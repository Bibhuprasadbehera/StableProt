#!/usr/bin/env python3
"""
Step 8: Final OGT evaluation and plotting.
Compares V7 (ensemble), PRIME, Ridge, XGBoost, and DPC baselines.
Generates: scatter_grid_ogt.png and temp_wise_ogt.png
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_FILE = PROJECT_ROOT / "data" / "embeddings" / "prepared_data_v7_saprot1.3b_seqonly_ogt_split.pt"
BASELINE_FILE = PROJECT_ROOT / "experiments" / "src" / "eval" / "ogt_baselines" / "prime_predictions.pt" # the one with PRIME added
V7_RESULTS_DIR = PROJECT_ROOT / "experiments" / "src" / "training" / "v7_shared" / "results"
OPTUNA_FILE = PROJECT_ROOT / "experiments" / "src" / "training" / "v7_shared" / "optuna_study" / "top3_configs.json"

import sys
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "src" / "training" / "v7_shared"))
from train import MultiHeadSaProtV7, SimpleDataset, CONFIG

def evaluate_metrics(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    pcc, _ = pearsonr(y_true, y_pred)
    spearman, _ = spearmanr(y_true, y_pred)
    print(f"[{name}] MAE: {mae:.3f} | RMSE: {rmse:.3f} | PCC: {pcc:.3f} | Spearman: {spearman:.3f}")
    return mae, rmse, pcc, spearman

def main():
    print("Loading data...")
    data = torch.load(DATA_FILE, map_location='cpu', weights_only=False)
    test_emb = data['test_ogt']['embeddings']
    y_true = data['test_ogt']['ogt_consensus']
    if isinstance(y_true, torch.Tensor): y_true = y_true.numpy()
    
    predictions = {}
    
    # 1. Load Baselines
    if BASELINE_FILE.exists():
        baselines = torch.load(BASELINE_FILE, map_location='cpu')
        for k, v in baselines.items():
            if k != 'y_true':
                predictions[k] = v
    else:
        # fallback if PRIME is still saving or running
        fallback = PROJECT_ROOT / "experiments" / "src" / "eval" / "ogt_baselines" / "baseline_predictions.pt"
        if fallback.exists():
            baselines = torch.load(fallback, map_location='cpu')
            for k, v in baselines.items():
                if k != 'y_true':
                    predictions[k] = v
        else:
            print("Baseline predictions not found!")
            
    # 2. V7 Ensemble Inference
    print("Running V7 Ensemble inference...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    loader = DataLoader(SimpleDataset(test_emb, torch.tensor(y_true)), batch_size=256, shuffle=False)
    
    v7_preds = []
    for seed in range(1, 6):
        model_path = V7_RESULTS_DIR / f"seed{seed}" / "best_model.pt"
        if not model_path.exists():
            continue
        model = MultiHeadSaProtV7(
            input_dim=1280, hidden1=CONFIG['hidden1'], hidden2=CONFIG['hidden2'],
            dropout1=CONFIG['dropout1'], dropout2=CONFIG['dropout2']
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        seed_preds = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                pred = model(x, task='ogt')
                seed_preds.extend(pred.cpu().numpy().tolist())
        v7_preds.append(seed_preds)
        
    if v7_preds:
        predictions['StableProt_V7'] = np.mean(v7_preds, axis=0)
        
    # 3. Print Metrics
    print("\n--- Final Test Set Results ---")
    metrics_log = {}
    for name, pred in predictions.items():
        metrics_log[name] = evaluate_metrics(y_true, pred, name)
        
    # 4. Plot Scatter Grid
    sns.set_theme(style="whitegrid", context="paper")
    models_to_plot = ['StableProt_V7', 'PRIME', 'XGBoost_SaProt', 'Ridge_SaProt', 'Ridge_DPC']
    models_to_plot = [m for m in models_to_plot if m in predictions]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, model_name in enumerate(models_to_plot):
        ax = axes[idx]
        y_p = predictions[model_name]
        
        ax.scatter(y_true, y_p, alpha=0.1, s=1, c='royalblue')
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        
        mae, rmse, pcc, spearman = metrics_log[model_name]
        text_str = f"MAE: {mae:.2f}°C\nPCC: {pcc:.2f}"
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(model_name.replace("_", " "))
        ax.set_xlabel("True OGT (°C)")
        ax.set_ylabel("Predicted OGT (°C)")
        
    # Hide unused subplots
    for idx in range(len(models_to_plot), len(axes)):
        axes[idx].set_visible(False)
        
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "scatter_grid_ogt.png", dpi=300)
    print("Saved scatter_grid_ogt.png")
    
    # 5. Plot Temperature-wise MAE
    bins = [0, 20, 30, 40, 50, 60, 70, 120]
    labels = ['<20', '20-30', '30-40', '40-50', '50-60', '60-70', '>70']
    
    binned_mae = {model: [] for model in models_to_plot}
    y_true_bins = np.digitize(y_true, bins) - 1
    
    for i in range(len(labels)):
        idx = (y_true_bins == i)
        for model in models_to_plot:
            if np.sum(idx) > 0:
                mae = mean_absolute_error(y_true[idx], predictions[model][idx])
            else:
                mae = 0
            binned_mae[model].append(mae)
            
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    width = 0.8 / len(models_to_plot)
    
    colors = sns.color_palette("husl", len(models_to_plot))
    for i, model in enumerate(models_to_plot):
        ax.bar(x + i*width - 0.4 + width/2, binned_mae[model], width, label=model.replace("_", " "), color=colors[i])
        
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(n={np.sum(y_true_bins==i)})" for i, l in enumerate(labels)])
    ax.set_xlabel("True OGT Range (°C)")
    ax.set_ylabel("Mean Absolute Error (°C)")
    ax.set_title("OGT Prediction Error Across Temperature Bins")
    ax.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "temp_wise_ogt.png", dpi=300)
    print("Saved temp_wise_ogt.png")

if __name__ == "__main__":
    main()
