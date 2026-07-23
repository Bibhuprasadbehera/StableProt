import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import lmdb
import glob
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score, f1_score
from sklearn.model_selection import KFold
from scipy.stats import spearmanr, pearsonr
from transformers import EsmTokenizer, EsmModel, AutoTokenizer, AutoModel, T5Tokenizer, T5EncoderModel

# Add training paths
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_file_dir, "../../.."))
v9_dir = os.path.join(project_root, "experiments/src/training/v9_disjoint")
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if v9_dir not in sys.path:
    sys.path.insert(0, v9_dir)

from train import MultiHeadSaProtV8, enrich_inputs
from inference.v7_predict import load_saprot_model

# Constants for normalization
ogt_mean, ogt_std = 37.51, 14.22
tm_mean, tm_std = 52.88, 16.50

def mask_sequence_for_saprot(seq: str) -> str:
    return "".join(f"{aa}#" for aa in seq)

def clean_seq(seq):
    return "".join([c for c in str(seq) if c.isupper() and c.isalpha()])

def get_saprot_embeddings_batched(model, tokenizer, sequences, device="cuda", batch_size=16):
    model.eval()
    all_embeddings = []
    
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        saprot_seqs = [mask_sequence_for_saprot(clean_seq(seq)) for seq in batch]
        
        inputs = tokenizer(
            saprot_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            hidden = outputs.last_hidden_state
            # Mean pool
            masked_hidden = hidden * attention_mask
            embs = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)
            all_embeddings.append(embs.cpu())
            
    return torch.cat(all_embeddings, dim=0)

def get_saprot_residue_embeddings_batched(model, tokenizer, sequences, device="cuda"):
    model.eval()
    all_res_embs = []
    for seq in sequences:
        cleaned = clean_seq(seq)
        L = len(cleaned)
        saprot_seq = mask_sequence_for_saprot(cleaned)
        inputs = tokenizer(
            [saprot_seq],
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # outputs.last_hidden_state is shape (1, L_tok, 1280)
            # Remove CLS and EOS tokens: 1 to L+1
            hidden = outputs.last_hidden_state[0, 1:L+1].cpu()
            all_res_embs.append(hidden)
    return all_res_embs

def get_esm2_embeddings_batched(model, tokenizer, sequences, device="cuda", batch_size=16):
    model.eval()
    all_embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        cleaned_batch = [clean_seq(seq) for seq in batch]
        inputs = tokenizer(cleaned_batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            hidden = outputs.last_hidden_state
            masked_hidden = hidden * attention_mask
            embs = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)
            all_embeddings.append(embs.float().cpu())
    return torch.cat(all_embeddings, dim=0)

def get_prott5_embeddings_batched(model, tokenizer, sequences, device="cuda", batch_size=8):
    model.eval()
    all_embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        processed = []
        for seq in batch:
            cleaned = clean_seq(seq).replace('U', 'X').replace('Z', 'X').replace('O', 'X').replace('B', 'X')
            spaced = " ".join(list(cleaned))
            processed.append(spaced)
        inputs = tokenizer(processed, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            hidden = outputs.last_hidden_state
            masked_hidden = hidden * attention_mask
            embs = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)
            all_embeddings.append(embs.float().cpu())
    return torch.cat(all_embeddings, dim=0)

def get_composition_embeddings(sequences):
    aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    embs = []
    for seq in sequences:
        cleaned = clean_seq(seq)
        counts = [cleaned.count(aa) for aa in aas]
        total = sum(counts)
        if total > 0:
            freqs = [c / total for c in counts]
        else:
            freqs = [0.0] * 20
        embs.append(freqs)
    return torch.tensor(embs, dtype=torch.float32)

def get_esm2_residue_embeddings_batched(model, tokenizer, sequences, device="cuda"):
    model.eval()
    all_res_embs = []
    for seq in sequences:
        cleaned = clean_seq(seq)
        inputs = tokenizer([cleaned], return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            L = len(cleaned)
            hidden = outputs.last_hidden_state[0, 1:L+1].float().cpu()
            all_res_embs.append(hidden)
    return all_res_embs

def get_prott5_residue_embeddings_batched(model, tokenizer, sequences, device="cuda"):
    model.eval()
    all_res_embs = []
    for seq in sequences:
        cleaned = clean_seq(seq).replace('U', 'X').replace('Z', 'X').replace('O', 'X').replace('B', 'X')
        spaced = " ".join(list(cleaned))
        inputs = tokenizer([spaced], return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            L = len(cleaned)
            hidden = outputs.last_hidden_state[0, :L].float().cpu()
            all_res_embs.append(hidden)
    return all_res_embs

def get_composition_residue_embeddings(sequences):
    aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    aa_to_idx = {aa: i for i, aa in enumerate(aas)}
    all_res_embs = []
    for seq in sequences:
        cleaned = clean_seq(seq)
        L = len(cleaned)
        res_embs = np.zeros((L, 20), dtype=np.float32)
        for j, aa in enumerate(cleaned):
            if aa in aa_to_idx:
                res_embs[j, aa_to_idx[aa]] = 1.0
        all_res_embs.append(torch.tensor(res_embs, dtype=torch.float32))
    return all_res_embs


def extract_v9_features_batched(models_tm, models_ogt, saprot_embs, sequences, device="cuda", batch_size=32):
    all_h2_ogt = []
    all_h2_tm = []
    all_preds = []
    
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i+batch_size]
        batch_embs = saprot_embs[i:i+batch_size].to(device)
        
        # 1. Stage 1: OGT
        emb_o, aux_o = enrich_inputs(batch_embs.cpu(), batch_seqs, tmhmm_flags=None, ogt_priors=None)
        emb_o = emb_o.to(device)
        aux_o = aux_o.to(device)
        
        h2_o_all = []
        ogt_preds_all = []
        with torch.no_grad():
            for m_o in models_ogt:
                h_aux_o = m_o.aux_proj_ogt(aux_o) if m_o.aux_proj_ogt is not None else aux_o
                h_o = torch.cat([emb_o, h_aux_o], dim=-1)
                h1_o = m_o.act(m_o.ln1_ogt(m_o.fc1_ogt(h_o)))
                if m_o.res_ogt is not None:
                    h2_o = m_o.act(m_o.ln2_ogt(m_o.fc2_ogt(h1_o)) + m_o.res_ogt(h1_o))
                else:
                    h2_o = m_o.act(m_o.ln2_ogt(m_o.fc2_ogt(h1_o)))
                ogt_pred_z = m_o.head_ogt(h2_o).squeeze(-1)
                h2_o_all.append(h2_o)
                ogt_preds_all.append(ogt_pred_z)
                
        h2_o_mean = torch.stack(h2_o_all, dim=0).mean(dim=0)
        ogt_preds_stack = torch.stack(ogt_preds_all, dim=0)
        ogt_mean_z = ogt_preds_stack.mean(dim=0)
        ogt_var_z = ogt_preds_stack.var(dim=0) if ogt_preds_stack.shape[0] > 1 else torch.zeros_like(ogt_mean_z)
        
        ogt_preds_np = (ogt_mean_z * ogt_std + ogt_mean).cpu().numpy()
        
        # 2. Stage 2: Tm using predicted OGT prior
        emb_t, aux_t = enrich_inputs(batch_embs.cpu(), batch_seqs, tmhmm_flags=None, ogt_priors=ogt_preds_np)
        emb_t = emb_t.to(device)
        aux_t = aux_t.to(device)
        
        h2_t_all = []
        tm_preds_all = []
        tm_vars_all = []
        with torch.no_grad():
            for m_t in models_tm:
                h_aux_t = m_t.aux_proj_tm(aux_t) if m_t.aux_proj_tm is not None else aux_t
                h_t = torch.cat([emb_t, h_aux_t], dim=-1)
                h1_t = m_t.act(m_t.ln1_tm(m_t.fc1_tm(h_t)))
                if m_t.res_tm is not None:
                    h2_t = m_t.act(m_t.ln2_tm(m_t.fc2_tm(h1_t)) + m_t.res_tm(h1_t))
                else:
                    h2_t = m_t.act(m_t.ln2_tm(m_t.fc2_tm(h1_t)))
                out_t = m_t.head_tm(h2_t)
                z_mean = out_t[:, 0]
                z_var = F.softplus(out_t[:, 1]) + 1e-4
                h2_t_all.append(h2_t)
                tm_preds_all.append(z_mean)
                tm_vars_all.append(z_var)
                
        h2_t_mean = torch.stack(h2_t_all, dim=0).mean(dim=0)
        tm_preds_stack = torch.stack(tm_preds_all, dim=0)
        tm_mean_z = tm_preds_stack.mean(dim=0)
        tm_var_z = torch.stack(tm_vars_all, dim=0).mean(dim=0)
        
        # Denormalize
        ogt_pred_final = ogt_mean_z * ogt_std + ogt_mean
        ogt_var_final = ogt_var_z * (ogt_std ** 2)
        tm_pred_final = tm_mean_z * tm_std + tm_mean
        tm_var_final = tm_var_z * (tm_std ** 2)
        
        predictions = torch.stack([tm_pred_final, tm_var_final, ogt_pred_final, ogt_var_final], dim=-1)
        
        all_h2_ogt.append(h2_o_mean.cpu())
        all_h2_tm.append(h2_t_mean.cpu())
        all_preds.append(predictions.cpu())
        
    return torch.cat(all_h2_ogt, dim=0), torch.cat(all_h2_tm, dim=0), torch.cat(all_preds, dim=0)

def extract_v9_residue_features_batched(models_tm, models_ogt, saprot_res_embs, sequences, device="cuda"):
    all_h2_res = []
    # For residue level, we project each residue embedding
    # Since V9 disjoint models are trained on mean-pooled embeddings, we can project the residue embeddings
    # by treating each residue as a mock sequence, or we can use the model weights to project the 1280-dim representation
    # to 256-dim tm_mlp and ogt_mlp.
    # Specifically, the mapping from 1280-dim embedding to h2 uses:
    # fc1_tm, fc2_tm, and res_tm, plus the projection layer.
    # To project residue embeddings, we can run them through the fc1 and fc2 layers with zero aux features!
    # This is a very clean way to get residue-level tm_mlp and ogt_mlp representations!
    for i, res_emb in enumerate(saprot_res_embs):
        L = res_emb.shape[0]
        seq = sequences[i]
        # Create zero aux features
        aux_o = torch.zeros((L, 8), device=device)
        aux_t = torch.zeros((L, 9), device=device)
        res_emb = res_emb.to(device)
        
        with torch.no_grad():
            # OGT MLP projection
            m_o = models_ogt[0] # use seed 1 for simplicity
            h_aux_o = m_o.aux_proj_ogt(aux_o) if m_o.aux_proj_ogt is not None else aux_o
            h_o = torch.cat([res_emb, h_aux_o], dim=-1)
            h1_o = m_o.act(m_o.ln1_ogt(m_o.fc1_ogt(h_o)))
            if m_o.res_ogt is not None:
                h2_o = m_o.act(m_o.ln2_ogt(m_o.fc2_ogt(h1_o)) + m_o.res_ogt(h1_o))
            else:
                h2_o = m_o.act(m_o.ln2_ogt(m_o.fc2_ogt(h1_o)))
                
            # Tm MLP projection
            m_t = models_tm[0]
            h_aux_t = m_t.aux_proj_tm(aux_t) if m_t.aux_proj_tm is not None else aux_t
            h_t = torch.cat([res_emb, h_aux_t], dim=-1)
            h1_t = m_t.act(m_t.ln1_tm(m_t.fc1_tm(h_t)))
            if m_t.res_tm is not None:
                h2_t = m_t.act(m_t.ln2_tm(m_t.fc2_tm(h1_t)) + m_t.res_tm(h1_t))
            else:
                h2_t = m_t.act(m_t.ln2_tm(m_t.fc2_tm(h1_t)))
                
        all_h2_res.append({
            "Raw SaProt": res_emb.cpu().numpy(),
            "StableProt-Tm MLP": h2_t.cpu().numpy(),
            "StableProt-OGT MLP": h2_o.cpu().numpy(),
            "StableProt-Combined": torch.cat([h2_t, h2_o], dim=-1).cpu().numpy()
        })
    return all_h2_res

class MLPProbe(nn.Module):
    def __init__(self, input_dim, output_dim, is_regression=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim)
        )
        self.is_regression = is_regression
        
    def forward(self, x):
        return self.net(x)

def train_eval_mlp_probe(X, y, is_regression=False, num_classes=1, epochs=80, lr=0.005, device="cuda"):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = torch.tensor(X[train_idx], dtype=torch.float32), torch.tensor(X[val_idx], dtype=torch.float32)
        
        if is_regression:
            y_train, y_val = torch.tensor(y[train_idx], dtype=torch.float32).unsqueeze(-1), torch.tensor(y[val_idx], dtype=torch.float32).unsqueeze(-1)
            criterion = nn.MSELoss()
        else:
            if num_classes > 2:
                y_train, y_val = torch.tensor(y[train_idx], dtype=torch.long), torch.tensor(y[val_idx], dtype=torch.long)
                criterion = nn.CrossEntropyLoss()
            else:
                y_train, y_val = torch.tensor(y[train_idx], dtype=torch.float32).unsqueeze(-1), torch.tensor(y[val_idx], dtype=torch.float32).unsqueeze(-1)
                criterion = nn.BCEWithLogitsLoss()
                
        model = MLPProbe(X.shape[1], num_classes if num_classes > 2 else 1, is_regression).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Train
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = model(X_train.to(device))
            loss = criterion(out, y_train.to(device))
            loss.backward()
            optimizer.step()
            
        # Eval
        model.eval()
        with torch.no_grad():
            preds = model(X_val.to(device)).cpu()
            if is_regression:
                r2 = r2_score(y_val.numpy(), preds.numpy())
                scores.append(r2)
            else:
                if num_classes > 2:
                    pred_classes = torch.argmax(preds, dim=-1).numpy()
                    acc = accuracy_score(y_val.numpy(), pred_classes)
                    scores.append(acc)
                else:
                    pred_probs = torch.sigmoid(preds).numpy()
                    pred_classes = (pred_probs > 0.5).astype(int)
                    acc = accuracy_score(y_val.numpy(), pred_classes)
                    scores.append(acc)
                    
    return np.mean(scores)

def evaluate_linear_probe(X, y, is_regression=False, num_classes=1):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        if is_regression:
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            r2 = r2_score(y_val, preds)
            scores.append(r2)
        else:
            model = LogisticRegression(max_iter=1000, C=1.0)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            acc = accuracy_score(y_val, preds)
            scores.append(acc)
            
    return np.mean(scores)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load SaProt embed model and tokenizer
    print("Loading SaProt baseline...")
    embed_model, tokenizer = load_saprot_model(device=device)

    # Load ESM-2 model and tokenizer
    print("Loading ESM-2 baseline...")
    esm2_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    esm2_model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)

    # Load ProtT5 model and tokenizer
    print("Loading ProtT5 baseline...")
    prott5_tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    prott5_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50").to(device).half()
    
    # Load V9 disjoint models
    print("Loading StableProt V9 disjoint models...")
    models_dir = "experiments/src/training/v9_disjoint/results"
    models_tm = []
    models_ogt = []
    for s in range(1, 6):
        pt_tm = os.path.join(models_dir, f"seed{s}/model_tm.pt")
        pt_ogt = os.path.join(models_dir, f"seed{s}/model_ogt.pt")
        if os.path.exists(pt_tm) and os.path.exists(pt_ogt):
            m_t = MultiHeadSaProtV8().to(device)
            m_t.load_state_dict(torch.load(pt_tm, map_location=device, weights_only=False))
            m_t.eval()
            models_tm.append(m_t)
            
            m_o = MultiHeadSaProtV8().to(device)
            m_o.load_state_dict(torch.load(pt_ogt, map_location=device, weights_only=False))
            m_o.eval()
            models_ogt.append(m_o)
            
    print(f"Loaded {len(models_tm)} seed models for StableProt V9.")
    
    results = {}
    
    # Helper function to extract all 5 features for a set of sequences
    def extract_all_representation_levels(sequences):
        saprot = get_saprot_embeddings_batched(embed_model, tokenizer, sequences, device=device, batch_size=16)
        ogt_mlp, tm_mlp, preds = extract_v9_features_batched(models_tm, models_ogt, saprot, sequences, device=device, batch_size=32)
        combined = torch.cat([tm_mlp, ogt_mlp], dim=-1)
        
        esm2 = get_esm2_embeddings_batched(esm2_model, esm2_tokenizer, sequences, device=device, batch_size=16)
        prott5 = get_prott5_embeddings_batched(prott5_model, prott5_tokenizer, sequences, device=device, batch_size=8)
        comp = get_composition_embeddings(sequences)
        
        return {
            "Raw SaProt": saprot.numpy(),
            "StableProt-Tm MLP": tm_mlp.numpy(),
            "StableProt-OGT MLP": ogt_mlp.numpy(),
            "StableProt-Combined": combined.numpy(),
            "StableProt-Predictions": preds.numpy(),
            "ESM-2 (650M)": esm2.numpy(),
            "ProtT5-XL": prott5.numpy(),
            "Composition": comp.numpy()
        }

    # Helper to evaluate all representation levels using Ridge/Logistic Regression and MLP
    def evaluate_all_features(X_dict, y, is_regression=False, num_classes=1):
        task_results = {}
        for rep_name, X in X_dict.items():
            print(f"  Evaluating {rep_name}...")
            lin_score = evaluate_linear_probe(X, y, is_regression, num_classes)
            mlp_score = train_eval_mlp_probe(X, y, is_regression, num_classes, device=device)
            task_results[rep_name] = {
                "Linear Probe": float(lin_score),
                "MLP Probe": float(mlp_score)
            }
        return task_results

    # -------------------------------------------------------------
    # TASK 1: Zero-Shot Fitness Prediction (ProteinGym)
    # -------------------------------------------------------------
    print("\nEvaluating Task 1: Zero-Shot Fitness Prediction (ProteinGym)...")
    pg_path = "data/emergent_benchmarks/DMS_substitutions.parquet"
    if os.path.exists(pg_path):
        df = pd.read_parquet(pg_path)
        dms_ids = df["DMS_id"].unique()[:20]
        spearmans_saprot = []
        spearmans_esm2 = []
        spearmans_prott5 = []
        spearmans_comp = []
        
        for dms_id in dms_ids:
            df_sub = df[df["DMS_id"] == dms_id].copy()
            if len(df_sub) < 50:
                continue
            wt_seq = df_sub["target_seq"].iloc[0]
            mut_seqs = df_sub["mutated_sequence"].tolist()
            scores = df_sub["DMS_score"].tolist()
            
            # Subsample to speed up
            if len(mut_seqs) > 100:
                indices = np.random.choice(len(mut_seqs), 100, replace=False)
                mut_seqs = [mut_seqs[idx] for idx in indices]
                scores = [scores[idx] for idx in indices]
                
            wt_emb = get_saprot_embeddings_batched(embed_model, tokenizer, [wt_seq], device=device, batch_size=1)
            mut_embs = get_saprot_embeddings_batched(embed_model, tokenizer, mut_seqs, device=device, batch_size=16)
            cos_sims = F.cosine_similarity(wt_emb, mut_embs, dim=-1).cpu().numpy()
            rho_saprot, _ = spearmanr(cos_sims, scores)
            if not np.isnan(rho_saprot):
                spearmans_saprot.append(rho_saprot)

            wt_esm2 = get_esm2_embeddings_batched(esm2_model, esm2_tokenizer, [wt_seq], device=device, batch_size=1)
            mut_esm2 = get_esm2_embeddings_batched(esm2_model, esm2_tokenizer, mut_seqs, device=device, batch_size=16)
            cos_esm2 = F.cosine_similarity(wt_esm2, mut_esm2, dim=-1).cpu().numpy()
            rho_esm2, _ = spearmanr(cos_esm2, scores)
            if not np.isnan(rho_esm2):
                spearmans_esm2.append(rho_esm2)

            wt_t5 = get_prott5_embeddings_batched(prott5_model, prott5_tokenizer, [wt_seq], device=device, batch_size=1)
            mut_t5 = get_prott5_embeddings_batched(prott5_model, prott5_tokenizer, mut_seqs, device=device, batch_size=8)
            cos_t5 = F.cosine_similarity(wt_t5, mut_t5, dim=-1).cpu().numpy()
            rho_t5, _ = spearmanr(cos_t5, scores)
            if not np.isnan(rho_t5):
                spearmans_prott5.append(rho_t5)

            wt_comp = get_composition_embeddings([wt_seq])
            mut_comp = get_composition_embeddings(mut_seqs)
            cos_comp = F.cosine_similarity(wt_comp, mut_comp, dim=-1).cpu().numpy()
            rho_comp, _ = spearmanr(cos_comp, scores)
            if not np.isnan(rho_comp):
                spearmans_comp.append(rho_comp)
                
        results["Task 1 (ProteinGym Zero-Shot Spearman rho)"] = {
            "Raw SaProt": float(np.mean(spearmans_saprot)) if spearmans_saprot else 0.0,
            "ESM-2 (650M)": float(np.mean(spearmans_esm2)) if spearmans_esm2 else 0.0,
            "ProtT5-XL": float(np.mean(spearmans_prott5)) if spearmans_prott5 else 0.0,
            "Composition": float(np.mean(spearmans_comp)) if spearmans_comp else 0.0
        }
        print("ProteinGym Zero-Shot Spearmans:", results["Task 1 (ProteinGym Zero-Shot Spearman rho)"])
    else:
        print("ProteinGym dataset not found, skipping.")

    # -------------------------------------------------------------
    # TASK 2: Protein-Protein Interaction (STRING Yeast/Human)
    # -------------------------------------------------------------
    print("\nEvaluating Task 2: PPI (HumanPPI)...")
    human_ppi_test = "data/emergent_benchmarks/HumanPPI/normal/test"
    if os.path.exists(human_ppi_test):
        env = lmdb.open(human_ppi_test, readonly=True)
        seqs_1 = []
        seqs_2 = []
        labels = []
        with env.begin() as txn:
            cursor = txn.cursor()
            for k, v in cursor:
                try:
                    data = json.loads(v.decode('utf-8'))
                    seqs_1.append(clean_seq(data['seq_1']))
                    seqs_2.append(clean_seq(data['seq_2']))
                    labels.append(int(data['label']))
                except Exception:
                    pass
                if len(seqs_1) >= 200:
                    break
        if seqs_1:
            # Extract features for both proteins and concatenate
            X_1 = extract_all_representation_levels(seqs_1)
            X_2 = extract_all_representation_levels(seqs_2)
            
            X_dict = {}
            for k in X_1.keys():
                # Concatenate features of the two proteins
                X_dict[k] = np.concatenate([X_1[k], X_2[k]], axis=-1)
                
            results["Task 2 (HumanPPI Accuracy)"] = evaluate_all_features(X_dict, np.array(labels), is_regression=False)
            print("Finished HumanPPI.")
    else:
        print("HumanPPI dataset not found, skipping.")

    # -------------------------------------------------------------
    # TASK 3: Subcellular Localization (DeepLoc)
    # -------------------------------------------------------------
    print("\nEvaluating Task 3: Subcellular Localization (DeepLoc cls2)...")
    deeploc_path = "data/emergent_benchmarks/DeepLoc/cls2/normal/test"
    if os.path.exists(deeploc_path):
        env = lmdb.open(deeploc_path, readonly=True)
        seqs = []
        labels = []
        with env.begin() as txn:
            cursor = txn.cursor()
            for k, v in cursor:
                try:
                    data = json.loads(v.decode('utf-8'))
                    seqs.append(clean_seq(data['seq']))
                    labels.append(int(data['label']))
                except Exception:
                    pass
                if len(seqs) >= 400:
                    break
        
        if seqs:
            X_dict = extract_all_representation_levels(seqs)
            results["Task 3 (DeepLoc cls2 Accuracy)"] = evaluate_all_features(X_dict, np.array(labels), is_regression=False)
            print("Finished DeepLoc cls2.")
    else:
        print("DeepLoc cls2 not found, skipping.")

    # -------------------------------------------------------------
    # TASK 4: Solubility (eSOL)
    # -------------------------------------------------------------
    print("\nEvaluating Task 4: Solubility (eSOL)...")
    esol_path = "data/emergent_benchmarks/eSOL/test.csv"
    if os.path.exists(esol_path):
        df = pd.read_csv(esol_path).dropna(subset=['aa_seq', 'label']).head(400)
        seqs = df['aa_seq'].tolist()
        labels = df['label'].tolist()
        
        X_dict = extract_all_representation_levels(seqs)
        results["Task 4 (eSOL R2)"] = evaluate_all_features(X_dict, np.array(labels), is_regression=True)
        print("Finished eSOL.")
    else:
        print("eSOL dataset not found, skipping.")

    # -------------------------------------------------------------
    # TASK 5: Enzyme Function (EC)
    # -------------------------------------------------------------
    print("\nEvaluating Task 5: Enzyme Function (EC)...")
    ec_path = "data/emergent_benchmarks/EC/AF2/normal/test"
    if os.path.exists(ec_path):
        env = lmdb.open(ec_path, readonly=True)
        seqs = []
        labels = []
        with env.begin() as txn:
            cursor = txn.cursor()
            for k, v in cursor:
                try:
                    data = json.loads(v.decode('utf-8'))
                    seqs.append(clean_seq(data['seq']))
                    labels.append(int(any(data['label'][i] for i in [459, 212, 400, 94, 270, 502, 14, 314, 258, 568])))
                except Exception:
                    pass
                if len(seqs) >= 400:
                    break
        
        if seqs:
            X_dict = extract_all_representation_levels(seqs)
            results["Task 5 (EC-1.x.x.x Binary Accuracy)"] = evaluate_all_features(X_dict, np.array(labels), is_regression=False)
            print("Finished EC.")
    else:
        print("EC dataset not found, skipping.")

    # -------------------------------------------------------------
    # TASK 6: Secondary Structure (CB513 Residue-Level)
    # -------------------------------------------------------------
    print("\nEvaluating Task 6: Secondary Structure (CB513 residue-level)...")
    cb513_path = "data/emergent_benchmarks/CB513/CB513.csv"
    if os.path.exists(cb513_path):
        df = pd.read_csv(cb513_path).dropna(subset=['input', 'dssp3']).head(50) # 50 sequences contains thousands of residues
        seqs = df['input'].tolist()
        labels_str = df['dssp3'].tolist()
        
        # Get residue-level embeddings
        res_embs_list = get_saprot_residue_embeddings_batched(embed_model, tokenizer, seqs, device=device)
        res_features_list = extract_v9_residue_features_batched(models_tm, models_ogt, res_embs_list, seqs, device=device)
        
        esm2_res_list = get_esm2_residue_embeddings_batched(esm2_model, esm2_tokenizer, seqs, device=device)
        prott5_res_list = get_prott5_residue_embeddings_batched(prott5_model, prott5_tokenizer, seqs, device=device)
        comp_res_list = get_composition_residue_embeddings(seqs)
        
        # Flatten all residues
        X_res = {
            "Raw SaProt": [], 
            "StableProt-Tm MLP": [], 
            "StableProt-OGT MLP": [], 
            "StableProt-Combined": [],
            "ESM-2 (650M)": [],
            "ProtT5-XL": [],
            "Composition": []
        }
        y_res = []
        
        dssp3_map = {'C': 0, 'H': 1, 'E': 2}
        
        for idx, feat_dict in enumerate(res_features_list):
            seq_len = len(seqs[idx])
            labels = [dssp3_map.get(c, 0) for c in labels_str[idx][:seq_len]]
            
            X_res["Raw SaProt"].append(feat_dict["Raw SaProt"][:len(labels)])
            X_res["StableProt-Tm MLP"].append(feat_dict["StableProt-Tm MLP"][:len(labels)])
            X_res["StableProt-OGT MLP"].append(feat_dict["StableProt-OGT MLP"][:len(labels)])
            X_res["StableProt-Combined"].append(feat_dict["StableProt-Combined"][:len(labels)])
            X_res["ESM-2 (650M)"].append(esm2_res_list[idx][:len(labels)].numpy())
            X_res["ProtT5-XL"].append(prott5_res_list[idx][:len(labels)].numpy())
            X_res["Composition"].append(comp_res_list[idx][:len(labels)].numpy())
            y_res.extend(labels)
            
        for k in X_res.keys():
            X_res[k] = np.concatenate(X_res[k], axis=0)
        y_res = np.array(y_res)
        
        # Subsample residues to 2000 for training speed
        if len(y_res) > 2000:
            indices = np.random.choice(len(y_res), 2000, replace=False)
            for k in X_res.keys():
                X_res[k] = X_res[k][indices]
            y_res = y_res[indices]
            
        results["Task 6 (CB513 dssp3 Residue Accuracy)"] = evaluate_all_features(X_res, y_res, is_regression=False, num_classes=3)
        print("Finished CB513.")
    else:
        print("CB513 dataset not found, skipping.")

    # -------------------------------------------------------------
    # TASK 7: Remote Homology (SCOP Fold Prediction)
    # -------------------------------------------------------------
    print("\nEvaluating Task 7: Remote Homology (SCOP)...")
    scop_path = "data/emergent_benchmarks/scop/test.parquet"
    if os.path.exists(scop_path):
        df = pd.read_parquet(scop_path).dropna(subset=['seq', 'label']).head(400)
        seqs = df['seq'].tolist()
        labels = df['label'].astype(int).tolist()
        
        unique_labels = sorted(list(set(labels)))
        label_map = {l: i for i, l in enumerate(unique_labels)}
        mapped_labels = [label_map[l] for l in labels]
        num_classes = len(unique_labels)
        
        X_dict = extract_all_representation_levels(seqs)
        results["Task 7 (SCOP Accuracy)"] = evaluate_all_features(X_dict, np.array(mapped_labels), is_regression=False, num_classes=num_classes)
        print("Finished SCOP.")
    else:
        print("SCOP dataset not found, skipping.")

    # -------------------------------------------------------------
    # TASK 8: LiveProteinBench (Post-2025 Tasks)
    # -------------------------------------------------------------
    print("\nEvaluating Task 8: LiveProteinBench Zero-Shot Correlations...")
    temp_json = "data/emergent_benchmarks/LiveProteinBench/dataset/QA/temperature.json"
    if os.path.exists(temp_json):
        try:
            with open(temp_json, 'r') as fh:
                data = json.load(fh)
            seqs = []
            labels = []
            for item in data[:200]:
                if "Protein Sequence" in item and "Answer Text" in item:
                    try:
                        val = float(item["Answer Text"])
                        seqs.append(clean_seq(item["Protein Sequence"]))
                        labels.append(val)
                    except ValueError:
                        pass
            
            if seqs:
                saprot = get_saprot_embeddings_batched(embed_model, tokenizer, seqs, device=device, batch_size=16)
                _, _, preds = extract_v9_features_batched(models_tm, models_ogt, saprot, seqs, device=device, batch_size=32)
                
                pred_tms = preds[:, 0].numpy()
                pred_ogts = preds[:, 2].numpy()
                
                r_tm, _ = pearsonr(pred_tms, labels)
                r_ogt, _ = pearsonr(pred_ogts, labels)
                
                results["Task 8 (LiveProteinBench Temperature Pearson r)"] = {
                    "Predicted Tm vs Actual Temp": float(r_tm),
                    "Predicted OGT vs Actual Temp": float(r_ogt)
                }
                print(f"LiveProteinBench Temp Pearson r: Tm={r_tm:.4f}, OGT={r_ogt:.4f}")
        except Exception as e:
            print("Error evaluating LiveProteinBench:", e)
    else:
        print("LiveProteinBench temperature task not found, skipping.")

    # Output results to JSON
    with open("data/emergent_benchmarks/evaluation_results.json", 'w') as fh:
        json.dump(results, fh, indent=2)
    print("\nAll evaluation probe results saved to data/emergent_benchmarks/evaluation_results.json")

if __name__ == "__main__":
    main()
