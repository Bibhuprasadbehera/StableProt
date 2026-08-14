#!/usr/bin/env python3
"""
SPURS & Megascale / FLIP Meltome Generalization Benchmark (`Benchmark 7`)

Evaluates StableProt V9 across 781 holdout test proteins from the FLIP Meltome / Megascale
thermostability benchmark to verify zero-shot / out-of-distribution absolute Tm prediction accuracy.

Outputs:
  - `spurs_megascale_summary.csv`
  - `spurs_megascale_scatter.png`
  - `spurs_megascale_scatter.json` (Universal JSON compliance)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
PROJECT_ROOT = os.path.dirname(EXPERIMENTS_DIR)
VERSION = os.environ.get("STABLEPROT_VERSION", "v9_disjoint")
sys.path.append(os.path.join(EXPERIMENTS_DIR, f"src/training/{VERSION}"))

from train import MultiHeadSaProtV8, enrich_inputs

OUT_DIR = os.path.join(PROJECT_ROOT, "paper/writeup/plots")
VAL_SUITE_DIR = os.path.join(EXPERIMENTS_DIR, "new_data/validation_suite")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(VAL_SUITE_DIR, exist_ok=True)

def load_v9_model(device):
    import inspect
    VERSION = os.environ.get("STABLEPROT_VERSION", "v9_disjoint")
    model_dir = os.path.join(EXPERIMENTS_DIR, f"src/training/{VERSION}/results/seed1")
    ckpt_path = os.path.join(model_dir, "model_tm.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    from config import CONFIG
    sig = inspect.signature(MultiHeadSaProtV8.__init__)
    model_kwargs = {'proj_dim': CONFIG.get('proj_dim', 64)}
    if 'use_residuals' in sig.parameters:
        model_kwargs['use_residuals'] = CONFIG.get('use_residuals', True)
    model = MultiHeadSaProtV8(**model_kwargs).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()
    return model

def main():
    print("=====================================================================================")
    print("  SPURS & MEGASCALE / FLIP MELTOME BENCHMARK (`Benchmark 7`)")
    print("=====================================================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    csv_path = os.path.join(PROJECT_ROOT, "data/flip_meltome/flip_clean.csv")
    embs_path = os.path.join(PROJECT_ROOT, "data/flip_meltome/flip_saprot_embs.pt")
    
    if not os.path.exists(csv_path) or not os.path.exists(embs_path):
        raise FileNotFoundError(f"Missing FLIP meltome files: {csv_path} or {embs_path}")
        
    df = pd.read_csv(csv_path)
    embs = torch.load(embs_path, map_location='cpu', weights_only=False)
    
    print(f"Loaded {len(df)} FLIP Meltome holdout sequences.")
    print(f"Embeddings tensor shape: {embs.shape}")
    
    seqs = df['sequence'].tolist()
    y_true = df['label'].values.astype(np.float32)
    
    # Check normalization stats
    VERSION = os.environ.get("STABLEPROT_VERSION", "v9_disjoint")
    stats_path = os.path.join(EXPERIMENTS_DIR, f"src/training/{VERSION}/results/normalization_stats.pt")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"normalization_stats.pt not found at {stats_path}")
    norms = torch.load(stats_path, map_location='cpu', weights_only=False)
    
    model = load_v9_model(device)
    
    # Enrich inputs with auxiliary features (using default OGT prior 50C for unseen external organisms)
    emb_t, aux_t = enrich_inputs(embs, seqs, tmhmm_flags=None, ogt_priors=[50.0]*len(seqs))
    
    preds = []
    with torch.no_grad():
        for i in range(0, len(emb_t), 256):
            mu_norm, _ = model(emb_t[i:i+256].to(device), aux_t[i:i+256].to(device), head='tm')
            preds.append(mu_norm.cpu().numpy() * norms['tm_std'] + norms['tm_mean'])
    y_pred = np.concatenate(preds).astype(np.float32)
    
    # Overall metrics
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    r_val, _ = pearsonr(y_true, y_pred)
    rho_val, _ = spearmanr(y_true, y_pred)
    
    print(f"\nOverall SPURS / Megascale Evaluation Results (N={len(y_true)}):")
    print(f"  MAE:       {mae:.4f}°C")
    print(f"  RMSE:      {rmse:.4f}°C")
    print(f"  Pearson r: {r_val:.4f}")
    print(f"  Spearman ρ:{rho_val:.4f}")
    
    # Stratified metrics
    masks = {
        "Mesophilic (<= 40°C)": y_true <= 40.0,
        "Moderate (40 - 60°C)": (y_true > 40.0) & (y_true <= 60.0),
        "Thermophilic (> 60°C)": y_true > 60.0
    }
    
    strat_results = []
    for regime, mask in masks.items():
        if np.sum(mask) > 0:
            m_mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
            m_rmse = np.sqrt(np.mean((y_true[mask] - y_pred[mask])**2))
            strat_results.append({
                "Regime": regime,
                "Sample_Count": int(np.sum(mask)),
                "MAE": round(float(m_mae), 4),
                "RMSE": round(float(m_rmse), 4)
            })
            print(f"  {regime:<22} (N={np.sum(mask):3d}): MAE = {m_mae:5.2f}°C | RMSE = {m_rmse:5.2f}°C")
            
    # Save CSV
    summary_df = pd.DataFrame([{
        "Dataset": "FLIP Meltome / Megascale Holdout",
        "Total_Samples": len(y_true),
        "Overall_MAE": round(float(mae), 4),
        "Overall_RMSE": round(float(rmse), 4),
        "Pearson_r": round(float(r_val), 4),
        "Spearman_rho": round(float(rho_val), 4)
    }])
    csv_out = os.path.join(VAL_SUITE_DIR, "spurs_megascale_summary.csv")
    summary_df.to_csv(csv_out, index=False)
    print(f"\nSaved summary CSV to: {csv_out}")
    
    # Export Universal JSON compliance
    json_out = os.path.join(OUT_DIR, "spurs_megascale_scatter.json")
    json_data = {
        "dataset": "FLIP Meltome / Megascale Holdout (N=781)",
        "overall_metrics": {
            "mae": float(mae),
            "rmse": float(rmse),
            "pearson_r": float(r_val),
            "spearman_rho": float(rho_val)
        },
        "stratified_metrics": strat_results,
        "coordinates": {
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
            "seqid": df['seqid'].tolist()
        }
    }
    with open(json_out, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved JSON plot data: {json_out}")
    
    # Plot Scatter Diagram
    sns.set_context("paper", font_scale=1.2)
    plt.figure(figsize=(7.5, 7.0))
    
    # Scatter colored by true regime
    colors_map = {
        "Mesophilic (<= 40°C)": "#3b82f6",
        "Moderate (40 - 60°C)": "#10b981",
        "Thermophilic (> 60°C)": "#ef4444"
    }
    for regime, mask in masks.items():
        if np.sum(mask) > 0:
            plt.scatter(y_true[mask], y_pred[mask], c=colors_map[regime], alpha=0.7, edgecolors='k', linewidth=0.5, s=45, label=f"{regime} ($N={np.sum(mask)}$)")
            
    # Diagonal parity and error bands
    min_v, max_v = min(np.min(y_true), np.min(y_pred)) - 5, max(np.max(y_true), np.max(y_pred)) + 5
    plt.plot([min_v, max_v], [min_v, max_v], 'k--', linewidth=1.5, label='Parity ($y = x$)')
    plt.fill_between([min_v, max_v], [min_v-5, max_v-5], [min_v+5, max_v+5], color='gray', alpha=0.15, label='$\pm 5^\circ\mathrm{C}$ Error Band')
    
    plt.xlim(min_v, max_v)
    plt.ylim(min_v, max_v)
    plt.xlabel("Experimental $T_m$ (°C)")
    plt.ylabel("StableProt V9 Predicted $T_m$ (°C)")
    plt.title(f"FLIP Meltome / Megascale Generalization\nMAE = {mae:.2f}°C, Pearson $r = {r_val:.2f}$ (`Benchmark 7`)")
    plt.legend(loc='upper left', framealpha=0.9)
    plt.tight_layout()
    
    p_out = os.path.join(OUT_DIR, "spurs_megascale_scatter.png")
    plt.savefig(p_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved parity scatter plot: {p_out}")

if __name__ == "__main__":
    main()
