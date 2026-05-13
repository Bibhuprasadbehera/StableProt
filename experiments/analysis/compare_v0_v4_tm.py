import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import sys

# Add v4 model path to sys.path to import architecture
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../v4_multihead")))
from model import MultiHead_TmPredictor
from config import CONFIG

def evaluate_v4(model_path, data_path, thresholds):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    data = torch.load(data_path, weights_only=True)
    test_tm = data['test_tm']
    train_tm = data['train_tm']
    
    x = test_tm['embeddings'].to(device)
    y_true = test_tm['labels'].numpy()
    
    # Load model
    model = MultiHead_TmPredictor(
        input_size=CONFIG['input_size'],
        hidden1=CONFIG['hidden_size_1'],
        hidden2=CONFIG['hidden_size_2']
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    tm_mean = train_tm['labels'].mean().item()
    tm_std = train_tm['labels'].std().item()
    
    with torch.no_grad():
        preds = model(x, head='tm').cpu()
        if CONFIG['target_normalization']:
            preds = preds * tm_std + tm_mean
            
    y_pred = preds.numpy().flatten()
    
    metrics_by_threshold = {}
    for t in thresholds:
        t_val = int(t[1:])
        y_true_bin = (y_true >= t_val).astype(int)
        y_pred_bin = (y_pred >= t_val).astype(int)
        
        if len(np.unique(y_true_bin)) > 1:
            auc = roc_auc_score(y_true_bin, y_pred)
        else:
            auc = 0.0
            
        metrics_by_threshold[t] = {
            "auc_roc": float(auc),
            "f1": float(f1_score(y_true_bin, y_pred_bin)),
            "mcc": float(matthews_corrcoef(y_true_bin, y_pred_bin)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "pcc": float(pearsonr(y_true, y_pred)[0])
        }
    
    return metrics_by_threshold

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    comp_json_path = os.path.join(base_dir, "../comparison/comparison.json")
    v4_model_path = os.path.join(base_dir, "../v4_multihead/results/best_model.pth")
    data_path = os.path.join(base_dir, "../../new_data/prepared_data_v2.pt")
    
    if not os.path.exists(comp_json_path):
        print(f"Error: {comp_json_path} not found.")
        return
        
    with open(comp_json_path, 'r') as f:
        comp_data = json.load(f)
        
    thresholds = [f"t{t}" for t in range(0, 101, 5)]
    print(f"Thresholds to evaluate: {thresholds}")
    
    print("Evaluating V4 model...")
    v4_metrics = evaluate_v4(v4_model_path, data_path, thresholds)
    
    # Ensure all thresholds exist in comp_data
    for t in thresholds:
        if t not in comp_data:
            comp_data[t] = {}
        comp_data[t]["v4_multihead"] = v4_metrics[t]
        
    # Save final comparison
    final_json_path = os.path.join(base_dir, "../comparison/comparison_v4.json")
    with open(final_json_path, 'w') as f:
        json.dump(comp_data, f, indent=2)
    print(f"Final comparison saved to {final_json_path}")
    
    # Plotting
    plot_comparison(comp_data, base_dir)

def plot_comparison(comp_data, base_dir):
    thresholds_int = sorted([int(t[1:]) for t in comp_data.keys()])
    # Pick a subset of thresholds for cleaner bars (every 10 degrees)
    plot_thresholds_int = [t for t in thresholds_int if t % 10 == 0]
    plot_thresholds = [f"t{t}" for t in plot_thresholds_int]
    
    models = ["v0_original", "v1_baseline", "v2_improved", "v4_multihead"]
    colors = ['#94a3b8', '#64748b', '#475569', '#3b82f6'] # Slate to Blue (V4 is bright blue)
    
    n_thresholds = len(plot_thresholds)
    n_models = len(models)
    x = np.arange(n_thresholds)
    bar_width = 0.8 / n_models
    
    # 1. AUC Grouped Bar Chart
    plt.figure(figsize=(14, 7))
    for i, model in enumerate(models):
        aucs = []
        for t in plot_thresholds:
            val = comp_data[t].get(model, {}).get("auc_roc", 0)
            aucs.append(val)
        
        plt.bar(x + i*bar_width - (0.8/2) + bar_width/2, aucs, 
                width=bar_width, label=model, color=colors[i], alpha=0.9, edgecolor='white')
        
    plt.title("AUC-ROC Comparison (Higher is Better)", fontsize=16, fontweight='bold')
    plt.xlabel("Temperature Threshold (°C)", fontsize=12, fontweight='bold')
    plt.ylabel("AUC-ROC Score", fontsize=12, fontweight='bold')
    plt.xticks(x, [f"{t}°C" for t in plot_thresholds_int], fontweight='bold')
    plt.ylim(0, 1.1)
    plt.legend(title="Model Version", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "../comparison/auc_comparison_v4.png"), dpi=300)
    
    # 2. F1 Grouped Bar Chart
    plt.figure(figsize=(14, 7))
    for i, model in enumerate(models):
        f1s = []
        for t in plot_thresholds:
            val = comp_data[t].get(model, {}).get("f1", 0)
            f1s.append(val)
            
        plt.bar(x + i*bar_width - (0.8/2) + bar_width/2, f1s, 
                width=bar_width, label=model, color=colors[i], alpha=0.9, edgecolor='white')
        
    plt.title("F1-Score Comparison (Higher is Better)", fontsize=16, fontweight='bold')
    plt.xlabel("Temperature Threshold (°C)", fontsize=12, fontweight='bold')
    plt.ylabel("F1-Score", fontsize=12, fontweight='bold')
    plt.xticks(x, [f"{t}°C" for t in plot_thresholds_int], fontweight='bold')
    plt.ylim(0, 1.1)
    plt.legend(title="Model Version", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "../comparison/f1_comparison_v4.png"), dpi=300)
    
    print("Plots saved to experiments/comparison/")

if __name__ == "__main__":
    main()
