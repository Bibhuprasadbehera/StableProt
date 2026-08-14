#!/usr/bin/env python3
"""
Run Mesophilic Downsampling Ablation (`Claim 1 & Claim 4`)

Trains identical MultiHeadSaProtV8 models across 5 retention rates:
  - 1.00 (100% - No subsampling)
  - 0.50 (50%)
  - 0.25 (25%)
  - 0.14 (14% - Default StableProt V8 sweet spot)
  - 0.05 (5%)

Evaluates on Mesophilic (20-40°C) and Thermophilic (>60°C) ProThermDB test data.
Generates dual Y-axis trade-off plot (`mesophilic_subsampling_ablation.png` + `.json`).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
PROJECT_ROOT = os.path.dirname(EXPERIMENTS_DIR)
sys.path.append(SCRIPT_DIR)
from train import MultiHeadSaProtV8, enrich_inputs, compute_bin_weights

OUT_DIR = os.path.join(PROJECT_ROOT, "paper/writeup/plots")
VAL_SUITE_DIR = os.path.join(EXPERIMENTS_DIR, "new_data/validation_suite")
ABL_DIR = os.path.join(SCRIPT_DIR, "results_ablation")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(VAL_SUITE_DIR, exist_ok=True)
os.makedirs(ABL_DIR, exist_ok=True)

def load_train_data_subsampled(rate, seed=42):
    train_path = os.path.join(PROJECT_ROOT, "data/embeddings/saprot_tm_struct_embeddings.pt")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data missing: {train_path}")
    data = torch.load(train_path, map_location='cpu', weights_only=False)
    tm_data = data['train_tm']
    
    X = tm_data['embeddings']
    y = tm_data['tm_consensus']
    seqs = tm_data['sequences']
    tmhmm = tm_data.get('tmhmm_tm_binary', [0]*len(y))
    ogts = tm_data.get('ogt', [50.0]*len(y))
    
    # Subsample mesophiles (<= 40°C)
    np.random.seed(seed)
    meso_idx = np.where(y <= 40.0)[0]
    thermo_idx = np.where(y > 40.0)[0]
    
    n_keep = int(len(meso_idx) * rate)
    kept_meso = np.random.choice(meso_idx, size=max(n_keep, 10), replace=False)
    final_idx = np.sort(np.concatenate([kept_meso, thermo_idx]))
    
    X_sub = X[final_idx]
    y_sub = y[final_idx]
    seqs_sub = [seqs[i] for i in final_idx]
    tmhmm_sub = [tmhmm[i] for i in final_idx]
    ogts_sub = [ogts[i] for i in final_idx]
    
    return X_sub, y_sub, seqs_sub, tmhmm_sub, ogts_sub

def train_ablation_model(rate, device, seed=42):
    model_dir = os.path.join(ABL_DIR, f"subsample_{rate:.2f}")
    os.makedirs(model_dir, exist_ok=True)
    ckpt_path = os.path.join(model_dir, "model_tm.pt")
    
    if os.path.exists(ckpt_path):
        print(f"  [Rate {rate:.2f}] Found cached checkpoint: {ckpt_path}")
        model = MultiHeadSaProtV8().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
        model.eval()
        return model
        
    print(f"  [Rate {rate:.2f}] Training fresh checkpoint on {device}...")
    X, y, seqs, tmhmm, ogts = load_train_data_subsampled(rate, seed)
    emb_t, aux_t = enrich_inputs(X, seqs, tmhmm_flags=tmhmm, ogt_priors=ogts)
    
    # Target normalization
    stats_path = os.path.join(SCRIPT_DIR, "results/normalization_stats.pt")
    if os.path.exists(stats_path):
        norms = torch.load(stats_path, map_location='cpu', weights_only=False)
        tm_mean, tm_std = norms['tm_mean'], norms['tm_std']
    else:
        tm_mean, tm_std = y.mean().item(), y.std().item()
        
    y_norm = (y - tm_mean) / tm_std
    
    # Train/val split (90/10)
    torch.manual_seed(seed)
    n_val = int(len(y) * 0.1)
    perm = torch.randperm(len(y))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    
    model = MultiHeadSaProtV8().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Compute weights once on raw labels
    bin_edges = np.arange(20.0, 101.0, 5.0)
    w_all = compute_bin_weights(y.numpy(), bin_edges)
    w_train = torch.tensor(w_all[train_idx.numpy()], dtype=torch.float32).to(device)
    w_val = torch.tensor(w_all[val_idx.numpy()], dtype=torch.float32).to(device)
    
    train_ds = TensorDataset(emb_t[train_idx], aux_t[train_idx], y_norm[train_idx], w_train.cpu())
    val_ds = TensorDataset(emb_t[val_idx], aux_t[val_idx], y_norm[val_idx], w_val.cpu())
    
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
    
    best_val_loss = float('inf')
    patience = 6
    patience_cnt = 0
    
    for epoch in range(1, 36):
        model.train()
        for bx, baux, by, bw in train_loader:
            bx, baux, by, bw = bx.to(device), baux.to(device), by.to(device), bw.to(device)
            optimizer.zero_grad()
            z_mu, z_var = model(bx, baux, head='tm')
            nll = 0.5 * (z_mu - by)**2 / z_var + 0.5 * torch.log(z_var)
            loss = (nll * bw).mean()
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, baux, by, bw in val_loader:
                bx, baux, by, bw = bx.to(device), baux.to(device), by.to(device), bw.to(device)
                z_mu, z_var = model(bx, baux, head='tm')
                nll = 0.5 * (z_mu - by)**2 / z_var + 0.5 * torch.log(z_var)
                val_loss += (nll * bw).sum().item()
        val_loss /= len(val_ds)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break
                
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()
    return model

def evaluate_on_protherm(model, device):
    # 1. Internal validation split (`val_tm`)
    train_path = os.path.join(PROJECT_ROOT, "data/embeddings/saprot_tm_struct_embeddings.pt")
    data_train = torch.load(train_path, map_location='cpu', weights_only=False)
    val_tm = data_train['val_tm']
    embs_val = val_tm['embeddings']
    seqs_val = val_tm['sequences']
    y_val = val_tm['tm_consensus'].numpy()
    
    # 2. ProTherm test embeddings (`protherm_v8_struct_embeddings.pt`)
    from Bio import SeqIO
    protherm_fasta = os.path.join(PROJECT_ROOT, 'new_data/prothermdb_validation.fasta')
    protherm_csv = os.path.join(PROJECT_ROOT, 'new_data/prothermdb_validation.csv')
    df_p = pd.read_csv(protherm_csv)
    protherm_dict = {str(row['UniProt_ID']): float(row['Tm']) for _, row in df_p.iterrows() if not np.isnan(row['Tm'])}
    
    protherm_emb_p = os.path.join(PROJECT_ROOT, "data/embeddings/protherm_v8_struct_embeddings.pt")
    p_data = torch.load(protherm_emb_p, map_location='cpu', weights_only=False)
    
    embs_p_list, seqs_p_list, y_p_list = [], [], []
    for record in SeqIO.parse(protherm_fasta, 'fasta'):
        seq = str(record.seq)
        uid = record.id.split('|')[0]
        if uid in protherm_dict and seq in p_data:
            embs_p_list.append(p_data[seq].cpu())
            seqs_p_list.append(seq)
            y_p_list.append(protherm_dict[uid])
            
    embs_p = torch.stack(embs_p_list)
    y_p = np.array(y_p_list)
    
    # 3. Combine all test samples (`val_tm + protherm`)
    embs_all = torch.cat([embs_val, embs_p], dim=0)
    seqs_all = seqs_val + seqs_p_list
    y_all = np.concatenate([y_val, y_p])
    
    emb_t, aux_t = enrich_inputs(embs_all, seqs_all, tmhmm_flags=None, ogt_priors=[50.0]*len(seqs_all))
    
    stats_path = os.path.join(SCRIPT_DIR, "results/normalization_stats.pt")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"normalization_stats.pt not found at {stats_path}")
    norms = torch.load(stats_path, map_location='cpu', weights_only=False)
    
    preds_list = []
    with torch.no_grad():
        for i in range(0, len(emb_t), 512):
            mu_norm, _ = model(emb_t[i:i+512].to(device), aux_t[i:i+512].to(device), head='tm')
            preds_list.append(mu_norm.cpu().numpy() * norms['tm_std'] + norms['tm_mean'])
    preds = np.concatenate(preds_list)
    
    mask_meso = (y_all >= 20.0) & (y_all <= 40.0)
    mask_thermo = y_all > 60.0
    
    mae_meso = np.mean(np.abs(y_all[mask_meso] - preds[mask_meso]))
    mae_thermo = np.mean(np.abs(y_all[mask_thermo] - preds[mask_thermo]))
    mae_overall = np.mean(np.abs(y_all - preds))
    
    return mae_meso, mae_thermo, mae_overall

def main():
    print("=====================================================================================")
    print("  MESOPHILIC SUBSAMPLING & TRADE-OFF ABLATION (`Claim 1 & Claim 4`)")
    print("=====================================================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    rates = [1.00, 0.50, 0.25, 0.14, 0.05]
    results = []
    
    for r in rates:
        print(f"\n── Evaluating Subsampling Retention Rate: {r*100:.0f}% ──")
        try:
            model = train_ablation_model(r, device)
            mae_m, mae_t, mae_o = evaluate_on_protherm(model, device)
            results.append({
                'Retention_Rate': r,
                'Retention_Pct': f"{r*100:.0f}%",
                'Mesophilic_MAE_20_40C': round(float(mae_m), 3),
                'Thermophilic_MAE_gt60C': round(float(mae_t), 3),
                'Overall_MAE': round(float(mae_o), 3)
            })
            print(f"  Result -> Meso MAE (20-40°C): {mae_m:.2f}°C | Thermo MAE (>60°C): {mae_t:.2f}°C")
        except Exception as e:
            print(f"  Error evaluating rate {r}: {e}")
            
    if not results:
        print("Error: Could not evaluate any ablation models.")
        return
        
    df_out = pd.DataFrame(results)
    csv_out = os.path.join(VAL_SUITE_DIR, "subsampling_ablation_results.csv")
    df_out.to_csv(csv_out, index=False)
    print(f"\nSaved ablation summary to: {csv_out}")
    
    # Export JSON coordinates (`Universal JSON compliance`)
    json_out = os.path.join(OUT_DIR, "mesophilic_subsampling_ablation.json")
    with open(json_out, "w") as f:
        json.dump(df_out.to_dict(orient="records"), f, indent=2)
    print(f"Saved JSON plot data: {json_out}")
    
    # Dual Y-axis Plot
    sns.set_context("paper", font_scale=1.2)
    fig, ax1 = plt.subplots(figsize=(8.5, 6))
    
    x = [d['Retention_Rate'] for d in results]
    y_thermo = [d['Thermophilic_MAE_gt60C'] for d in results]
    y_meso = [d['Mesophilic_MAE_20_40C'] for d in results]
    
    color_t = '#ef4444' # Red for thermophilic
    ax1.set_xlabel("Mesophilic Retention Rate ($1.0 = 100\%$, $0.14 = 14\%$ Default)")
    ax1.set_ylabel("Thermophilic $T_m$ MAE ($>60^\circ$C, Lower is Better)", color=color_t, fontweight='bold')
    l1 = ax1.plot(x, y_thermo, 'o-', color=color_t, linewidth=2.5, markersize=8, label='Thermophilic MAE ($>60^\circ$C)')
    ax1.tick_params(axis='y', labelcolor=color_t)
    ax1.invert_xaxis() # Show 1.0 (100%) down to 0.05 (5%) on right
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    ax2 = ax1.twinx()
    color_m = '#2563eb' # Blue for mesophilic
    ax2.set_ylabel("Mesophilic $T_m$ MAE ($20-40^\circ$C, Lower is Better)", color=color_m, fontweight='bold')
    l2 = ax2.plot(x, y_meso, 's--', color=color_m, linewidth=2.5, markersize=8, label='Mesophilic MAE ($20-40^\circ$C)')
    ax2.tick_params(axis='y', labelcolor=color_m)
    
    # Mark sweet spot at 0.14
    ax1.axvline(x=0.14, color='#10b981', linestyle='-.', linewidth=2, label='Optimal Sweet Spot ($14\%$ retention)')
    
    lines = l1 + l2 + [plt.Line2D([0], [0], color='#10b981', linestyle='-.', linewidth=2)]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', framealpha=0.9)
    
    plt.title("Trade-off Analysis: Mesophilic Subsampling vs. Thermophilic Generalization")
    plt.tight_layout()
    p_out = os.path.join(OUT_DIR, "mesophilic_subsampling_ablation.png")
    plt.savefig(p_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved dual Y-axis plot: {p_out}")

if __name__ == "__main__":
    main()
