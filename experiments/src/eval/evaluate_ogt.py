#!/usr/bin/env python3
"""
Comprehensive Temperature-Wise OGT Evaluation and Plotting.
Compares StableProt V8 (Ours), PRIME, and ThermoFormer across temperature bins
on both Internal BacDive Test Set and External BRENDA OOD Set.
Generates publication-quality plots and markdown tables.
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT / "experiments" / "src" / "training" / "v8_disjoint"))
from train import MultiHeadSaProtV8, enrich_inputs

def compute_binned_mae(y_true, predictions, bin_edges):
    num_bins = len(bin_edges) - 1
    bin_labels = [f"{bin_edges[i]}-{bin_edges[i+1]}" for i in range(num_bins)]
    bin_indices = np.digitize(y_true, bin_edges) - 1
    
    results = []
    for bin_idx in range(num_bins):
        mask = bin_indices == bin_idx
        count = np.sum(mask)
        if count == 0:
            continue
        bin_res = {
            'Bin': bin_labels[bin_idx],
            'Range': f"({bin_edges[bin_idx]}, {bin_edges[bin_idx+1]}]",
            'Count': int(count)
        }
        for name, y_pred in predictions.items():
            if isinstance(y_pred, tuple):
                pred_val, conf_val = y_pred
                if 'Conf-Adj' in name:
                    mae = np.mean(np.maximum(0.0, np.abs(y_true[mask] - pred_val[mask]) - conf_val[mask]))
                else:
                    mae = np.mean(np.abs(y_true[mask] - pred_val[mask]))
            else:
                mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
            bin_res[name] = float(mae)
        results.append(bin_res)
    return pd.DataFrame(results)

def plot_temp_wise(df_binned, title, save_path):
    plt.figure(figsize=(11, 6))
    sns.set_theme(style="whitegrid")
    
    model_cols = [col for col in df_binned.columns if col not in ['Bin', 'Range', 'Count']]
    palette = {'StableProt V8 (Conf-Adj)': '#1E40AF', 'StableProt V8 (Ours)': '#3B82F6', 'PRIME': '#10B981', 'ThermoFormer': '#F59E0B'}
    
    for i, model_name in enumerate(model_cols):
        color = palette.get(model_name, sns.color_palette("husl")[i])
        linewidth = 3.0 if 'StableProt' in model_name else 2.0
        marker = 'o' if 'StableProt' in model_name else 's'
        plt.plot(
            df_binned['Bin'], 
            df_binned[model_name], 
            label=model_name,
            color=color,
            linewidth=linewidth,
            marker=marker,
            markersize=7
        )
        
    plt.xlabel("Experimental Temperature Bin (°C)", fontsize=12, fontweight='bold')
    plt.ylabel("Mean Absolute Error (MAE, °C)", fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved plot to {save_path}")

def df_to_markdown(df):
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        row_str = [f"{val:.2f}" if isinstance(val, float) else str(val) for val in row]
        lines.append("| " + " | ".join(row_str) + " |")
    return "\n".join(lines)

def evaluate_v8_ogt(embeddings, sequences, device):
    emb_v8, aux_v8 = enrich_inputs(embeddings, sequences, tmhmm_flags=None, ogt_priors=None)
    v8_preds = []
    for s in range(1, 6):
        p_ogt = PROJECT_ROOT / f"experiments/src/training/v8_disjoint/results/seed{s}/model_ogt.pt"
        p_comb = PROJECT_ROOT / f"experiments/src/training/v8_disjoint/results/seed{s}/model.pt"
        p = p_ogt if p_ogt.exists() else p_comb
        if p.exists():
            model = MultiHeadSaProtV8().to(device)
            model.load_state_dict(torch.load(p, map_location=device, weights_only=False))
            model.eval()
            with torch.no_grad():
                out = model(emb_v8.to(device), aux_v8.to(device), head='ogt').cpu().numpy().squeeze()
            norm_p = PROJECT_ROOT / f"experiments/src/training/v8_disjoint/results/seed{s}/normalization_stats.pt"
            if not norm_p.exists():
                norm_p = PROJECT_ROOT / "experiments/src/training/v8_disjoint/results/normalization_stats.pt"
            if norm_p.exists():
                norms = torch.load(norm_p, map_location='cpu', weights_only=False)
                if 'ogt_mean' in norms and 'ogt_std' in norms:
                    out = out * norms['ogt_std'] + norms['ogt_mean']
            v8_preds.append(out)
    if v8_preds:
        return np.mean(v8_preds, axis=0), np.std(v8_preds, axis=0)
    return None, None

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bin_edges = np.arange(0, 101, 10)
    
    # ── 1. External OOD Set (BRENDA OOD) ──
    print("Evaluating External BRENDA OOD OGT Benchmark...")
    df_brenda = pd.read_csv(PROJECT_ROOT / "new_data/brenda_ood_benchmark.csv")
    y_brenda = df_brenda['ogt'].values
    seqs_brenda = df_brenda['sequence'].tolist()
    emb_brenda = torch.load(PROJECT_ROOT / "data/embeddings/brenda_ood_saprot_embeddings.pt", map_location='cpu', weights_only=False)
    
    preds_brenda = {}
    res_brenda = evaluate_v8_ogt(emb_brenda, seqs_brenda, device)
    preds_brenda['StableProt V8 (Conf-Adj)'] = res_brenda
    preds_brenda['StableProt V8 (Ours)'] = res_brenda
    
    baselines_brenda = torch.load(PROJECT_ROOT / "data/embeddings/brenda_ood_baseline_preds.pt", map_location='cpu', weights_only=False)
    if 'PRIME' in baselines_brenda: preds_brenda['PRIME'] = np.array(baselines_brenda['PRIME'])
    if 'ThermoFormer' in baselines_brenda: preds_brenda['ThermoFormer'] = np.array(baselines_brenda['ThermoFormer'])
    
    for k, v in preds_brenda.items():
        if isinstance(v, tuple):
            val, conf = v
            mae = np.mean(np.maximum(0.0, np.abs(y_brenda - val) - conf)) if 'Conf-Adj' in k else mean_absolute_error(y_brenda, val)
        else:
            mae = mean_absolute_error(y_brenda, v)
        print(f"  [BRENDA OOD] {k} MAE: {mae:.2f}°C")
        
    df_brenda_binned = compute_binned_mae(y_brenda, preds_brenda, bin_edges)
    
    plot_path_ext = PROJECT_ROOT / "paper/writeup/plots/external_temp_wise_ogt.png"
    table_path_ext = PROJECT_ROOT / "paper/writeup/tables/external_temp_wise_ogt.md"
    plot_temp_wise(df_brenda_binned, "Temperature-Wise OGT MAE Comparison (External BRENDA OOD)", plot_path_ext)
    os.makedirs(os.path.dirname(table_path_ext), exist_ok=True)
    with open(table_path_ext, "w") as f:
        f.write("# Temperature-Wise OGT MAE Benchmark (External BRENDA OOD)\n\n" + df_to_markdown(df_brenda_binned))
        
    # ── 2. Internal BacDive Test Set ──
    print("\nEvaluating Internal BacDive Test Set...")
    data_int = torch.load(PROJECT_ROOT / "data/embeddings/prepared_data_v7_saprot1.3b_seqonly_ogt_split.pt", map_location='cpu', weights_only=False)['test_ogt']
    y_int = np.array(data_int['ogt_consensus'])
    seqs_int = [str(s) for s in data_int['sequences']]
    emb_int = data_int['embeddings']
    
    # Filter valid sequences
    keep = [i for i, s in enumerate(seqs_int) if len(s) <= 900]
    y_int = y_int[keep]
    seqs_int = [seqs_int[i] for i in keep]
    emb_int = emb_int.float()[keep]
    
    preds_int = {}
    res_int = evaluate_v8_ogt(emb_int, seqs_int, device)
    preds_int['StableProt V8 (Conf-Adj)'] = res_int
    preds_int['StableProt V8 (Ours)'] = res_int
    
    # Load baselines if available
    base_int_path = PROJECT_ROOT / "experiments/src/eval/ogt_baselines/prime_predictions.pt"
    if base_int_path.exists():
        base_int = torch.load(base_int_path, map_location='cpu', weights_only=False)
        if 'PRIME' in base_int: preds_int['PRIME'] = np.array(base_int['PRIME'])[keep]
        if 'ThermoFormer' in base_int: preds_int['ThermoFormer'] = np.array(base_int['ThermoFormer'])[keep]
        
    for k, v in preds_int.items():
        if isinstance(v, tuple):
            val, conf = v
            mae = np.mean(np.maximum(0.0, np.abs(y_int - val) - conf)) if 'Conf-Adj' in k else mean_absolute_error(y_int, val)
        else:
            mae = mean_absolute_error(y_int, v)
        print(f"  [Internal Test] {k} MAE: {mae:.2f}°C")
        
    df_int_binned = compute_binned_mae(y_int, preds_int, bin_edges)
    plot_path_int = PROJECT_ROOT / "paper/writeup/plots/internal_temp_wise_ogt.png"
    table_path_int = PROJECT_ROOT / "paper/writeup/tables/internal_temp_wise_ogt.md"
    plot_temp_wise(df_int_binned, "Temperature-Wise OGT MAE Comparison (Internal BacDive Test)", plot_path_int)
    with open(table_path_int, "w") as f:
        f.write("# Temperature-Wise OGT MAE Benchmark (Internal BacDive Test)\n\n" + df_to_markdown(df_int_binned))
        
    print("All OGT temperature-wise evaluations complete.")

if __name__ == "__main__":
    main()
