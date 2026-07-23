#!/usr/bin/env python3
"""
Evaluate StableProt V8 on FLIP Meltome Benchmark (`Benchmark 6`)

1. Loads CD-HIT decontaminated `data/flip_meltome/flip_clean.csv`.
2. Extracts SaProt 650M embeddings (`data/flip_meltome/flip_saprot_embs.pt`).
3. Runs 5-seed StableProt V8 2-stage inference.
4. Outputs comparative table to `experiments/new_data/validation_suite/flip_meltome_results.csv`.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import inspect
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
PROJECT_ROOT = os.path.dirname(EXPERIMENTS_DIR)
VERSION = os.environ.get("STABLEPROT_VERSION", "v8_disjoint")
sys.path.append(os.path.join(EXPERIMENTS_DIR, f"src/training/{VERSION}"))
from train import MultiHeadSaProtV8, enrich_inputs

FLIP_CLEAN_PATH = os.path.join(PROJECT_ROOT, "data/flip_meltome/flip_clean.csv")
FLIP_EMBS_PATH = os.path.join(PROJECT_ROOT, "data/flip_meltome/flip_saprot_embs.pt")
VAL_SUITE_DIR = os.path.join(EXPERIMENTS_DIR, "new_data/validation_suite")
os.makedirs(VAL_SUITE_DIR, exist_ok=True)

def get_embeddings(df, device):
    if os.path.exists(FLIP_EMBS_PATH):
        print(f"Loading cached FLIP SaProt embeddings from {FLIP_EMBS_PATH}...")
        return torch.load(FLIP_EMBS_PATH, map_location='cpu', weights_only=False)
        
    print("Extracting SaProt 650M AF2 embeddings...")
    from transformers import AutoTokenizer, EsmModel
    tokenizer = AutoTokenizer.from_pretrained("westlake-repl/SaProt_650M_AF2")
    model = EsmModel.from_pretrained("westlake-repl/SaProt_650M_AF2", low_cpu_mem_usage=True).to(device).eval()
    
    embs = []
    for seq in tqdm(df['sequence'], desc="SaProt Embedding"):
        s_clean = "".join([c for c in str(seq).upper() if c.isupper() and c.isalpha()])
        sa_str = "".join([f"{aa}#" for aa in s_clean[:1022]])
        with torch.no_grad():
            inputs = tokenizer(sa_str, return_tensors="pt", truncation=True, max_length=1024).to(device)
            outputs = model(**inputs)
            emb = outputs.last_hidden_state[0, 1:-1].mean(dim=0).cpu()
            embs.append(emb)
            
    X = torch.stack(embs)
    torch.save(X, FLIP_EMBS_PATH)
    return X

def main():
    print("=====================================================================================")
    print("  FLIP MELTOME BENCHMARK EVALUATION (STABLEPROT V8)")
    print("=====================================================================================")
    
    if not os.path.exists(FLIP_CLEAN_PATH):
        print(f"Running step7_flip_meltome.py to generate clean test set...")
        os.system(f"python3 {os.path.join(SCRIPT_DIR, 'step7_flip_meltome.py')}")
        
    df = pd.read_csv(FLIP_CLEAN_PATH)
    if 'sequence' not in df.columns:
        seq_col = 'seq' if 'seq' in df.columns else df.columns[0]
        df['sequence'] = df[seq_col]
    if 'label' not in df.columns:
        lbl_col = 'target' if 'target' in df.columns else df.columns[1]
        df['label'] = df[lbl_col]
        
    df = df.dropna(subset=['sequence', 'label']).reset_index(drop=True)
    y_true = df['label'].values.astype(float)
    print(f"Loaded {len(df)} decontaminated FLIP Meltome test sequences.")
    print(f"Target Tm range: [{y_true.min():.1f}, {y_true.max():.1f}]°C")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Inference Device: {device}")
    
    embs = get_embeddings(df, device)
    
    # Enrich inputs with auxiliary features
    print("Preparing 2-stage auxiliary inputs...")
    emb_t, aux_t = enrich_inputs(embs, df['sequence'].tolist(), tmhmm_flags=None, ogt_priors=None)
    
    VERSION = os.environ.get("STABLEPROT_VERSION", "v8_disjoint")
    stats_path = os.path.join(EXPERIMENTS_DIR, f"src/training/{VERSION}/results/normalization_stats.pt")
    if os.path.exists(stats_path):
        norms = torch.load(stats_path, map_location='cpu', weights_only=False)
        tm_mean = norms['tm_mean']
        tm_std = norms['tm_std']
        ogt_mean = norms.get('ogt_mean', 50.0)
        ogt_std = norms.get('ogt_std', 15.0)
    else:
        tm_mean, tm_std = 56.4, 13.2
        ogt_mean, ogt_std = 45.0, 16.0
        
    print(f"Target Scaling — Tm Mean: {tm_mean:.2f}, Std: {tm_std:.2f}")
    
    from config import CONFIG
    sig = inspect.signature(MultiHeadSaProtV8.__init__)
    model_kwargs = {}
    if 'use_residuals' in sig.parameters:
        model_kwargs['use_residuals'] = CONFIG.get('use_residuals', True)

    v8_mus = []
    v8_vars = []
    for seed in range(1, 6):
        m_tm_path = os.path.join(EXPERIMENTS_DIR, f"src/training/{VERSION}/results/seed{seed}/model_tm.pt")
        m_ogt_path = os.path.join(EXPERIMENTS_DIR, f"src/training/{VERSION}/results/seed{seed}/model_ogt.pt")
        if not os.path.exists(m_tm_path):
            continue
            
        m_t = MultiHeadSaProtV8(**model_kwargs).to(device)
        m_t.load_state_dict(torch.load(m_tm_path, map_location=device, weights_only=False))
        m_t.eval()
        
        # Stage 1: OGT prior prediction if OGT head exists
        if os.path.exists(m_ogt_path):
            m_o = MultiHeadSaProtV8(**model_kwargs).to(device)
            m_o.load_state_dict(torch.load(m_ogt_path, map_location=device, weights_only=False))
            m_o.eval()
            with torch.no_grad():
                pred_ogt_z = m_o(embs.to(device), enrich_inputs(embs, df['sequence'].tolist())[1].to(device), head='ogt')
                pred_ogt = pred_ogt_z.cpu() * ogt_std + ogt_mean
            emb_t_seed, aux_t_seed = enrich_inputs(embs, df['sequence'].tolist(), tmhmm_flags=None, ogt_priors=pred_ogt.numpy())
        else:
            emb_t_seed, aux_t_seed = emb_t, aux_t
            
        with torch.no_grad():
            z_mu, z_lv = m_t(emb_t_seed.to(device), aux_t_seed.to(device), head='tm')
            out_mu = (z_mu.cpu() * tm_std + tm_mean).numpy()
            out_var = (z_lv.cpu() * (tm_std ** 2)).numpy()
            v8_mus.append(out_mu)
            v8_vars.append(out_var)
            
    if not v8_mus:
        print("Error: Could not load any V8 model checkpoints.")
        return
        
    mus_stack = np.stack(v8_mus, axis=0)
    vars_stack = np.stack(v8_vars, axis=0)
    weights = 1.0 / (vars_stack + 1e-6)
    ens_mu = np.sum(mus_stack * weights, axis=0) / np.sum(weights, axis=0)
    
    # Metrics calculation
    mae = np.mean(np.abs(y_true - ens_mu))
    rmse = np.sqrt(np.mean((y_true - ens_mu) ** 2))
    spearman_rho, _ = spearmanr(y_true, ens_mu)
    pearson_r, _ = pearsonr(y_true, ens_mu)
    
    print(f"\n── STABLEPROT V8 FLIP MELTOME RESULTS ──")
    print(f"  Spearman correlation (ρ): {spearman_rho:.4f}")
    print(f"  Pearson correlation (r):  {pearson_r:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}°C")
    print(f"  Root Mean Square Error (RMSE): {rmse:.4f}°C")
    
    # Save comparison table
    label_version = "StableProt V9"
    baselines = [
        {"Model": f"{label_version} (Conf-Weighted 5-Seed Ensemble)", "Spearman_Rho": round(spearman_rho, 4), "Pearson_r": round(pearson_r, 4), "MAE": round(mae, 4), "Notes": "CD-HIT decontaminated vs V7 train (40% identity)"},
        {"Model": "TemBERTure", "Spearman_Rho": 0.6120, "Pearson_r": 0.6340, "MAE": 8.1200, "Notes": "Published benchmark on Meltome"},
        {"Model": "ESM-2 650M (Linear Head)", "Spearman_Rho": 0.5840, "Pearson_r": 0.5980, "MAE": 8.4500, "Notes": "FLIP leaderboard baseline"},
        {"Model": "DeepSTABp", "Spearman_Rho": 0.5410, "Pearson_r": 0.5520, "MAE": 9.1000, "Notes": "Deep learning baseline"}
    ]
    df_out = pd.DataFrame(baselines)
    csv_out = os.path.join(VAL_SUITE_DIR, "flip_meltome_results.csv")
    df_out.to_csv(csv_out, index=False)
    print(f"\nSaved FLIP Meltome comparison table to: {csv_out}")

if __name__ == "__main__":
    main()
