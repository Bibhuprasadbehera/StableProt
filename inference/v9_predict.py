import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import EsmTokenizer, EsmModel

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
v8_dir = os.path.join(root_dir, "experiments/src/training/v8_disjoint")
if v8_dir not in sys.path:
    sys.path.insert(0, v8_dir)

from train import MultiHeadSaProtV8, enrich_inputs
from inference.v7_predict import load_saprot_model, get_saprot_embedding

class V9Predictor:
    def __init__(self, models_dir: str, device="cuda"):
        self.device = device
        self.embed_model, self.tokenizer = load_saprot_model(device=device)
        self.models_tm = []
        self.models_ogt = []
        
        # Load normalization stats
        norm_path = os.path.join(models_dir, "normalization_stats.pt")
        if os.path.exists(norm_path):
            norms = torch.load(norm_path, map_location='cpu', weights_only=False)
            self.tm_mean = norms.get('tm_mean', 52.88)
            self.tm_std = norms.get('tm_std', 16.50)
            self.ogt_mean = norms.get('ogt_mean', 37.51)
            self.ogt_std = norms.get('ogt_std', 14.22)
            print(f"Loaded normalization stats: Tm=N({self.tm_mean:.2f}, {self.tm_std:.2f}), OGT=N({self.ogt_mean:.2f}, {self.ogt_std:.2f})")
        else:
            self.tm_mean, self.tm_std = 52.88, 16.50
            self.ogt_mean, self.ogt_std = 37.51, 14.22
            print("WARNING: normalization_stats.pt not found. Using default stats.")
            
        for s in range(1, 6):
            pt_tm = os.path.join(models_dir, f"seed{s}/model_tm.pt")
            pt_ogt = os.path.join(models_dir, f"seed{s}/model_ogt.pt")
            if os.path.exists(pt_tm) and os.path.exists(pt_ogt):
                m_t = MultiHeadSaProtV8().to(device)
                m_t.load_state_dict(torch.load(pt_tm, map_location=device, weights_only=False))
                m_t.eval()
                self.models_tm.append(m_t)
                
                m_o = MultiHeadSaProtV8().to(device)
                m_o.load_state_dict(torch.load(pt_ogt, map_location=device, weights_only=False))
                m_o.eval()
                self.models_ogt.append(m_o)
                
        if not self.models_tm:
            raise FileNotFoundError(f"No V9 seed models found in {models_dir}")
        print(f"Loaded {len(self.models_tm)} seed models for V9 ensemble.")

    def predict(self, sequence: str):
        emb = get_saprot_embedding(self.embed_model, self.tokenizer, sequence, device=self.device)
        emb = emb.float()
        
        with torch.no_grad():
            # 1. Stage 1: Predict OGT across 5 seeds
            emb_o, aux_o = enrich_inputs(emb.cpu(), [sequence], tmhmm_flags=None, ogt_priors=None)
            ogt_preds = []
            for m_o in self.models_ogt:
                pred_z = m_o(emb_o.to(self.device), aux_o.to(self.device), head='ogt')
                ogt_preds.append(pred_z.item())
                
            ogt_preds = np.array(ogt_preds)
            ogt_mu_z = np.mean(ogt_preds)
            ogt_sigma_z = np.std(ogt_preds) if len(ogt_preds) > 1 else 0.5  # Epistemic uncertainty across seeds
            
            # Denormalize OGT
            ogt_val = ogt_mu_z * self.ogt_std + self.ogt_mean
            ogt_conf = ogt_sigma_z * self.ogt_std
            
            # 2. Stage 2: Predict Tm using predicted OGT prior
            emb_t, aux_t = enrich_inputs(emb.cpu(), [sequence], tmhmm_flags=None, ogt_priors=np.array([ogt_val]))
            mus = []
            vars_list = []
            for m_t in self.models_tm:
                z_mu, z_lv = m_t(emb_t.to(self.device), aux_t.to(self.device), head='tm')
                pred_mu = z_mu.cpu() * self.tm_std + self.tm_mean
                pred_var = z_lv.cpu() * (self.tm_std ** 2)
                mus.append(pred_mu)
                vars_list.append(pred_var)
                
            mus_stack = torch.stack(mus, dim=0)
            vars_stack = torch.stack(vars_list, dim=0)
            
            # Confidence-weighted ensemble: weight by inverse variance
            weights = 1.0 / (vars_stack + 1e-6)
            tm_val = (mus_stack * weights).sum(dim=0) / weights.sum(dim=0)
            total_var = 1.0 / weights.sum(dim=0)
            tm_conf = torch.sqrt(total_var)
            
        return tm_val.item(), tm_conf.item(), ogt_val, ogt_conf
