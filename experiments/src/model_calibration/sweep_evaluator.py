#!/usr/bin/env python3
"""
Isolated Model Calibration Sweep Evaluator (`experiments/src/model_calibration/sweep_evaluator.py`)

Evaluates all sweep runs from `experiments/src/model_calibration/logs/summary_metrics.csv`:
- Computes full per-bin OGT MAE (`0-10, 10-20, ..., 90-100°C`)
- Computes overall OGT MAE and Spearman rank correlation (`rho`)
- Generates JSON + PNG grids cleanly under `experiments/src/model_calibration/plots/`
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT / "experiments" / "src" / "model_calibration"))
from sweep_runner import CalibrationSaProtV8, enrich_inputs

def evaluate_checkpoint_ogt(model_path, data_ogt_split, ogt_mean, ogt_std, device):
    if not os.path.exists(model_path):
        return None
    model = CalibrationSaProtV8().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    
    seqs = data_ogt_split['sequences']
    embs = data_ogt_split['embeddings']
    lbls = data_ogt_split.get('ogt_consensus', data_ogt_split.get('labels'))
    tmhmm = data_ogt_split.get('tmhmm_tm_binary', [0]*len(seqs))
    
    e_t, aux_t = enrich_inputs(embs, seqs, tmhmm)
    
    preds_list = []
    targets_list = []
    with torch.no_grad():
        for i in range(0, len(e_t), 64):
            eb = e_t[i:i+64].to(device)
            ab = aux_t[i:i+64].to(device)
            po = model(eb, ab, head='ogt')
            po_raw = po * ogt_std + ogt_mean
            preds_list.append(po_raw.cpu().numpy())
            targets_list.append([float(l) for l in lbls[i:i+64]])
            
    preds = np.concatenate(preds_list)
    targets = np.concatenate(targets_list)
    
    overall_mae = np.mean(np.abs(preds - targets))
    rho, _ = spearmanr(preds, targets)
    r, _ = pearsonr(preds, targets)
    
    # 10°C per-bin MAE (`0-10, 10-20, ..., 90-100`)
    bin_edges = list(range(0, 110, 10))
    bin_maes = {}
    for b_idx in range(len(bin_edges)-1):
        low, high = bin_edges[b_idx], bin_edges[b_idx+1]
        mask = (targets >= low) & (targets < high)
        if np.sum(mask) > 0:
            bin_maes[f"{low}-{high}°C"] = float(np.mean(np.abs(preds[mask] - targets[mask])))
        else:
            bin_maes[f"{low}-{high}°C"] = None
            
    return {
        'overall_ogt_mae': float(overall_mae),
        'spearman_rho': float(rho),
        'pearson_r': float(r),
        'per_bin_mae': bin_maes
    }

def main():
    parser = argparse.ArgumentParser(description="Isolated Model Calibration Sweep Evaluator")
    parser.add_argument("--metrics_file", type=str, default="experiments/src/model_calibration/logs/summary_metrics.csv")
    parser.add_argument("--output_dir", type=str, default="experiments/src/model_calibration/plots")
    args = parser.parse_args()

    metrics_path = PROJECT_ROOT / args.metrics_file
    if not metrics_path.exists():
        print(f"No summary CSV found at {metrics_path}. Run sweep_runner.py first.")
        return

    df = pd.read_csv(metrics_path)
    print(f"Loaded {len(df)} runs from {metrics_path}")

    out_dir = PROJECT_ROOT / args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ogt_split = torch.load("data/embeddings/prepared_data_v7_saprot1.3b_seqonly_ogt_split.pt", map_location='cpu', weights_only=False)['val_ogt']

    # Process each run and plot trends
    for g in df['group'].unique():
        sub_g = df[df['group'] == g]
        for p in sub_g['param'].unique():
            sub_p = sub_g[sub_g['param'] == p].drop_duplicates(subset=['value'], keep='last').sort_values(by='value')
            if len(sub_p) == 0:
                continue

            x_vals = [str(v) for v in sub_p['value']]
            y_tm = sub_p['best_val_tm_mae'].values
            y_ogt = sub_p['best_val_ogt_mae'].values

            fig, ax1 = plt.subplots(figsize=(8, 5))
            ax2 = ax1.twinx()

            l1 = ax1.plot(x_vals, y_tm, marker='o', color='#2b5c8f', label='Val $T_m$ MAE (°C)', linewidth=2)
            l2 = ax2.plot(x_vals, y_ogt, marker='s', color='#d95f02', label='Val OGT MAE (°C)', linewidth=2, linestyle='--')

            ax1.set_xlabel(f"Calibration Parameter: {p}", fontsize=11, fontweight='bold')
            ax1.set_ylabel("$T_m$ MAE (°C)", color='#2b5c8f', fontsize=11, fontweight='bold')
            ax2.set_ylabel("OGT MAE (°C)", color='#d95f02', fontsize=11, fontweight='bold')
            ax1.tick_params(axis='y', labelcolor='#2b5c8f')
            ax2.tick_params(axis='y', labelcolor='#d95f02')

            plt.title(f"Model Calibration Sweep: [{g}] | {p}", fontsize=13, fontweight='bold')
            plt.grid(True, linestyle=':', alpha=0.6)

            lns = l1 + l2
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs, loc='upper right')

            plt.tight_layout()
            plot_name = f"calib_sweep_{g}_{p}.png"
            json_name = f"calib_sweep_{g}_{p}.json"
            plt.savefig(out_dir / plot_name, dpi=300)
            plt.close()

            manifest = {
                'group': g,
                'param': p,
                'values': list(x_vals),
                'val_tm_mae': list([float(v) for v in y_tm]),
                'val_ogt_mae': list([float(v) for v in y_ogt]),
                'runtimes_sec': list([float(v) for v in sub_p['runtime_sec'].values])
            }
            with open(out_dir / json_name, 'w') as jf:
                json.dump(manifest, jf, indent=2)

            print(f"  Exported calibration plot -> {out_dir / plot_name}")

if __name__ == "__main__":
    main()
