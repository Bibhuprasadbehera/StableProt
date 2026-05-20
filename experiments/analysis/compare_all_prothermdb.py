"""
Ultimate Grand Unified ProThermDB Benchmark.

Evaluates all model iterations (V0 to V6) directly on the gold-standard
experimental melting temperature (Tm) test set from ProThermDB.

Models compared:
  - V0 Original (Pre-trained binary ensemble, ProtT5)
  - V1 Baseline (Retrained binary ensemble, ProtT5)
  - V2 Improved (Specialized binary ensemble, ProtT5)
  - V3 Regression (Single-head continuous OGT regression, ProtT5)
  - V4 Improved Regression (Residual single-head OGT regression, ProtT5)
  - V5 Multi-Head ProtT5 (Shared backbone, distinct Tm head)
  - V6 Multi-Head ESM-2 (Shared backbone, distinct Tm head, ESM-2 3B)

Generates:
  - Premium grouped performance bar charts contrasting MAE and RMSE side-by-side.
  - Multi-panel grid of predicted vs. true Tm scatter plots with correlation overlays.
  - Stratified violin error distributions illustrating zero-mean prediction stability.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(EXPERIMENTS_DIR)

# ── Self-contained model definitions to prevent module cache side-effects ──

class MLP_C2H2(nn.Module):
    def __init__(self, input_size=1024, hidden_size_1=512, hidden_size_2=256):
        super().__init__()
        self.model = nn.ModuleList([
            nn.Linear(input_size, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, 1),
            nn.Sigmoid()
        ])
    def forward(self, point):
        for layer in self.model:
            point = layer(point)
        return point

class MLP_Baseline(nn.Module):
    def __init__(self, input_size=1024, hidden_size_1=256, hidden_size_2=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

class MLP_Improved(nn.Module):
    def __init__(self, input_size=1024, hidden_size_1=512, hidden_size_2=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.BatchNorm1d(hidden_size_1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.BatchNorm1d(hidden_size_2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size_2, 1)
        )
    def forward(self, x):
        return self.model(x)

class MLP_Regression(nn.Module):
    def __init__(self, input_size=1024, hidden_size_1=512, hidden_size_2=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.BatchNorm1d(hidden_size_1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.BatchNorm1d(hidden_size_2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size_2, 1)
        )
    def forward(self, x):
        return self.model(x)

class MLP_Regression_Improved(nn.Module):
    def __init__(self, input_size=1024, hidden_size_1=512, hidden_size_2=256):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.bn1 = nn.BatchNorm1d(hidden_size_1)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.bn2 = nn.BatchNorm1d(hidden_size_2)
        self.dropout2 = nn.Dropout(0.2)
        self.residual_proj = nn.Linear(hidden_size_1, hidden_size_2) if hidden_size_1 != hidden_size_2 else nn.Identity()
        self.head = nn.Linear(hidden_size_2, 1)
    def forward(self, x):
        x1 = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x2 = self.dropout2(F.relu(self.bn2(self.fc2(x1)) + self.residual_proj(x1)))
        return self.head(x2)

def load_v0_model(model_path, device='cpu'):
    state_dict = torch.load(model_path, map_location=torch.device(device), weights_only=False)['state_dict']
    new_state_dict = {}
    for key in list(state_dict.keys()):
        new_key = key.replace('model.model.', 'model.')
        new_state_dict[new_key] = state_dict[key]
    h1 = new_state_dict['model.0.weight'].shape[0]
    h2 = new_state_dict['model.2.weight'].shape[0]
    input_size = new_state_dict['model.0.weight'].shape[1]
    classifier = MLP_C2H2(input_size=input_size, hidden_size_1=h1, hidden_size_2=h2)
    classifier.load_state_dict(new_state_dict)
    classifier.eval()
    classifier.to(device)
    return classifier

def compute_expected_temperatures(prob_matrix, thresholds):
    step_sizes = np.diff(thresholds)
    step = step_sizes[0] if len(step_sizes) > 0 else 5
    base_temp = max(0, thresholds[0] - step)
    y_pred = np.full(prob_matrix.shape[0], base_temp, dtype=float)
    for i in range(len(thresholds)):
        y_pred += prob_matrix[:, i] * step
    return y_pred

def compute_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    pcc, _ = scipy.stats.pearsonr(y_true, y_pred)
    spearman, _ = scipy.stats.spearmanr(y_true, y_pred)
    r2 = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
    
    # Binary classification metrics at standard thermostability threshold (>= 60°C)
    y_true_b = (y_true >= 60.0).astype(int)
    y_pred_b = (y_pred >= 60.0).astype(int)
    
    tp = np.sum((y_true_b == 1) & (y_pred_b == 1))
    fp = np.sum((y_true_b == 0) & (y_pred_b == 1))
    fn = np.sum((y_true_b == 1) & (y_pred_b == 0))
    tn = np.sum((y_true_b == 0) & (y_pred_b == 0))
    
    acc = (tp + tn) / len(y_true_b) if len(y_true_b) > 0 else 0.0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    
    denom = np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    mcc = (float(tp * tn) - float(fp * fn)) / denom if denom > 0 else 0.0
    
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1.0))) * 100.0
    
    k = max(1, int(0.1 * len(y_true)))
    t_top10 = np.percentile(y_true, 90)
    top_k_indices = np.argsort(y_pred)[-k:]
    enrich = np.sum(y_true[top_k_indices] >= t_top10) / float(k)
    
    try:
        from sklearn.metrics import roc_auc_score
        roc_auc = roc_auc_score(y_true_b, y_pred)
    except Exception:
        roc_auc = 0.5 + 0.5 * (sens + spec - 1.0)
    
    # Multi-threshold ROC AUC (biologically meaningful boundaries)
    from sklearn.metrics import roc_auc_score as _roc_auc
    threshold_aucs = {}
    for t in [45, 55, 65, 80]:
        y_bin = (y_true >= t).astype(int)
        if len(np.unique(y_bin)) > 1:
            try:
                threshold_aucs[f'roc_auc_{t}C'] = _roc_auc(y_bin, y_pred)
            except Exception:
                threshold_aucs[f'roc_auc_{t}C'] = 0.5
        else:
            threshold_aucs[f'roc_auc_{t}C'] = float('nan')
    
    # Global AUC: mean ROC AUC across thresholds 30-90°C
    global_aucs = []
    for t in range(30, 91):
        y_bin = (y_true >= t).astype(int)
        if len(np.unique(y_bin)) > 1:
            try:
                global_aucs.append(_roc_auc(y_bin, y_pred))
            except Exception:
                pass
    global_auc = np.mean(global_aucs) if global_aucs else 0.5
        
    return {
        'mae': mae, 'rmse': rmse, 'pcc': pcc, 'spearman': spearman, 'r2': r2,
        'acc': acc, 'sens': sens, 'spec': spec, 'prec': prec, 'f1': f1, 'mcc': mcc,
        'mape': mape, 'enrich': enrich, 'roc_auc': roc_auc,
        **threshold_aucs, 'global_auc': global_auc
    }

def main():
    print("=" * 85)
    print("  ULTIMATE GRAND UNIFIED BENCHMARK: EXPERIMENTAL MELTING TEMPERATURE (ProThermDB)")
    print("=" * 85)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing evaluations on device: {device}")
    
    # Load ProThermDB test datasets
    prott5_data_path = os.path.join(PROJECT_ROOT, "new_data/prepared_data_v5_prott5.pt")
    esm2_data_path = os.path.join(PROJECT_ROOT, "new_data/prepared_data_v2.pt")
    
    if not os.path.exists(prott5_data_path) or not os.path.exists(esm2_data_path):
        print(f"Error: Prepared datasets missing. Check paths:\n  {prott5_data_path}\n  {esm2_data_path}")
        return
        
    d_prott5 = torch.load(prott5_data_path, weights_only=False)
    x_prott5 = d_prott5['test_tm']['embeddings'].to(device)
    y_true_prott5 = d_prott5['test_tm']['labels'].numpy()
    
    results = {}
    

    # ── 1. TemStaPro (V0 Original) ──
    print("\nEvaluating TemStaPro (V0 Original)...")
    v0_thresholds = [40, 45, 50, 55, 60, 65]
    v0_models_dir = os.path.join(PROJECT_ROOT, "StableProt/models")
    v0_probs = []
    for t in v0_thresholds:
        t_probs = []
        for s in range(1, 6):
            p = os.path.join(v0_models_dir, f"mean_major_imbal-{t}_s{s}.pt")
            if os.path.exists(p):
                model = load_v0_model(p, device=device)
                with torch.no_grad():
                    out = model(x_prott5.float()).squeeze().cpu().numpy()
                t_probs.append(out)
        if t_probs:
            v0_probs.append(np.mean(t_probs, axis=0))
        else:
            v0_probs.append(np.zeros(len(y_true_prott5)))
    v0_preds = compute_expected_temperatures(np.column_stack(v0_probs), v0_thresholds)
    results['TemStaPro (V0 Original)'] = {'y_true': y_true_prott5, 'y_pred': v0_preds, 'type': 'Binary Proxy'}

    # ── 2. V2 Improved ──
    print("Evaluating V2 Improved...")
    v2_thresholds = list(range(5, 100, 5))
    v2_probs = []
    for t in v2_thresholds:
        t_probs = []
        for s in range(1, 6):
            p = os.path.join(EXPERIMENTS_DIR, f"v2_improved/results/t{t}/seed{s}/model.pt")
            if os.path.exists(p):
                model = MLP_Improved().to(device)
                model.load_state_dict(torch.load(p, map_location=device, weights_only=False))
                model.eval()
                with torch.no_grad():
                    logits = model(x_prott5.float()).squeeze()
                    out = torch.sigmoid(logits).cpu().numpy()
                t_probs.append(out)
        if t_probs:
            v2_probs.append(np.mean(t_probs, axis=0))
        else:
            v2_probs.append(np.zeros(len(y_true_prott5)))
    v2_preds = compute_expected_temperatures(np.column_stack(v2_probs), v2_thresholds)
    results['V2 Improved'] = {'y_true': y_true_prott5, 'y_pred': v2_preds, 'type': 'Binary Proxy'}

    # ── 3. V3 Regression ──
    print("Evaluating V3 Regression...")
    v3_preds = []
    for s in range(1, 6):
        p = os.path.join(EXPERIMENTS_DIR, f"v3_regression/results/seed{s}/model.pt")
        if os.path.exists(p):
            model = MLP_Regression().to(device)
            model.load_state_dict(torch.load(p, map_location=device, weights_only=False))
            model.eval()
            with torch.no_grad():
                out = model(x_prott5.float()).squeeze().cpu().numpy()
            v3_preds.append(out)
    if v3_preds:
        results['V3 Regression'] = {'y_true': y_true_prott5, 'y_pred': np.mean(v3_preds, axis=0), 'type': 'Continuous Proxy'}

    # ── 4. V4 Improved Regression ──
    print("Evaluating V4 Improved Regression...")
    v4_preds = []
    for s in range(1, 6):
        p = os.path.join(EXPERIMENTS_DIR, f"v4_improved/results/seed{s}/model.pt")
        if os.path.exists(p):
            model = MLP_Regression_Improved().to(device)
            model.load_state_dict(torch.load(p, map_location=device, weights_only=False))
            model.eval()
            with torch.no_grad():
                out = model(x_prott5.float()).squeeze().cpu().numpy()
            v4_preds.append(out)
    if v4_preds:
        results['V4 Improved Regr.'] = {'y_true': y_true_prott5, 'y_pred': np.mean(v4_preds, axis=0), 'type': 'Continuous Proxy'}

    # ── 5. TemBERTure & ESMStabP ──
    # [REMOVED]: Synthetic generation of literature baselines was removed to ensure scientific integrity.
    # True benchmarking requires running the actual models on this specific holdout set.

    # ── 7. V5 Multi-Head ProtT5 ──
    print("Loading V5 Multi-Head predictions...")
    v5_p = os.path.join(EXPERIMENTS_DIR, "v5_multihead/results/ensemble/predictions.pt")
    if os.path.exists(v5_p):
        d = torch.load(v5_p, weights_only=False)
        results['V5 Multi-Head (ProtT5)'] = {
            'y_true': d['y_true'].numpy() if hasattr(d['y_true'], 'numpy') else np.array(d['y_true']),
            'y_pred': d['y_pred'].numpy() if hasattr(d['y_pred'], 'numpy') else np.array(d['y_pred']),
            'type': 'Dedicated Tm Head'
        }

    # ── 8. V6 Multi-Head ESM-2 ──
    print("Loading V6 Multi-Head predictions...")
    v6_p = os.path.join(EXPERIMENTS_DIR, "v6_multihead_esm2/results/ensemble/predictions.pt")
    if os.path.exists(v6_p):
        d = torch.load(v6_p, weights_only=False)
        results['V6 Multi-Head (ESM-2)'] = {
            'y_true': d['y_true'].numpy() if hasattr(d['y_true'], 'numpy') else np.array(d['y_true']),
            'y_pred': d['y_pred'].numpy() if hasattr(d['y_pred'], 'numpy') else np.array(d['y_pred']),
            'type': 'Dedicated Tm Head'
        }

    # ── Display Summary Metrics Table ──
    print("\nFINAL PROTHERMDB EXPERIMENTAL TM PREDICTION BENCHMARK (Advanced Robustness & Biophysical Enrichment Suite):")
    print("-" * 175)
    print(f"{'Model Iteration':<25} | {'Type':<18} | {'MAE':<6} | {'PCC':<5} | {'R²':<6} | {'MCC':<6} | {'F1':<5} | {'AUC':<5} | {'MAPE(%)':<8} | {'Top-10% Enrich':<14}")
    print("-" * 175)
    
    metrics_summary = {}
    for name, data in results.items():
        m = compute_metrics(data['y_true'], data['y_pred'])
        metrics_summary[name] = m
        print(f"{name:<25} | {data['type']:<18} | {m['mae']:<6.2f} | {m['pcc']:<5.2f} | {m['r2']:<6.2f} | {m['mcc']:<6.3f} | {m['f1']:<5.2f} | {m['roc_auc']:<5.2f} | {m['mape']:<8.1f} | {m['enrich']:<14.3f}")
    print("-" * 175)

    # ==========================================
    # PUBLICATION MASTERPIECE VISUALIZATIONS
    # ==========================================
    output_dir = os.path.join(SCRIPT_DIR, 'prothermdb_comparison')
    os.makedirs(output_dir, exist_ok=True)
    
    # Premium curated, scientifically harmonious colors palette
    colors = ['#64748b', '#3b82f6', '#0284c7', '#f59e0b', '#d97706', '#10b981', '#059669']
    
    sns.set_context("paper", font_scale=1.25)
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        pass
        
    # ── Plot 1: Premium Grouped Performance Bar Chart ──
    plt.figure(figsize=(15, 7))
    models = list(results.keys())
    n_models = len(models)
    x = np.arange(n_models)
    width = 0.35
    
    maes = [metrics_summary[m]['mae'] for m in models]
    rmses = [metrics_summary[m]['rmse'] for m in models]
    
    bars1 = plt.bar(x - width/2, maes, width, label='MAE (°C)', color='#3b82f6', alpha=0.95, edgecolor='none', zorder=3)
    bars2 = plt.bar(x + width/2, rmses, width, label='RMSE (°C)', color='#f59e0b', alpha=0.95, edgecolor='none', zorder=3)
    
    plt.ylabel('Error in Degrees Celsius (°C)', fontweight='bold', fontsize=13)
    plt.title('Prediction Error Profiles on ProThermDB Experimental Melting Temperatures', fontweight='bold', fontsize=15, pad=15)
    plt.xticks(x, models, rotation=20, ha='right', fontweight='bold', fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    plt.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=12)
    
    # Annotate values clearly on premium bars
    for bar in bars1:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold', fontsize=10, color='#1e293b')
    for bar in bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold', fontsize=10, color='#1e293b')
        
    plt.ylim(0, max(rmses) * 1.15)
    plt.tight_layout()
    bar_path = os.path.join(output_dir, 'overall_metrics_barplot.png')
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # ── Plot 2: Grid of Comparative Scatter Subplots ──
    cols = 3
    rows = int(np.ceil(n_models / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4.5 * rows))
    axes = axes.flatten()
    
    for i, name in enumerate(models):
        ax = axes[i]
        d = results[name]
        m = metrics_summary[name]
        
        # Dense scatter display using custom premium alpha and compact point scaling
        ax.scatter(d['y_true'], d['y_pred'], alpha=0.25, color=colors[i % len(colors)], s=12, edgecolor='none')
        ax.plot([0, 105], [0, 105], color='#ef4444', linestyle='--', linewidth=2, alpha=0.85)
        
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
        ax.set_title(f"{name}\nMAE: {m['mae']:.2f}°C | PCC: {m['pcc']:.2f}", fontweight='bold', fontsize=12)
        ax.set_xlabel('Experimental Tm (°C)', fontweight='bold', fontsize=11)
        ax.set_ylabel('Predicted Tm (°C)', fontweight='bold', fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.5)
        
    # Hide unused grid subplots
    for j in range(n_models, len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, 'comparative_scatter_grid.png')
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # ── Plot 3: Premium Stratified Violin Error Distribution ──
    plt.figure(figsize=(14, 6))
    plot_errors = [results[m]['y_pred'] - results[m]['y_true'] for m in models]
    
    parts = plt.violinplot(plot_errors, showmeans=True, showmedians=False, showextrema=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_edgecolor('none')
        pc.set_alpha(0.7)
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
        vp = parts[partname]
        vp.set_edgecolor('#334155')
        vp.set_linewidth(1.2)
        
    plt.axhline(0, color='#ef4444', linestyle='--', linewidth=1.5, alpha=0.8, label='Zero Error Baseline')
    plt.xticks(np.arange(1, n_models + 1), models, rotation=15, ha='right', fontweight='bold', fontsize=11)
    plt.ylabel('Prediction Error Profile (Pred - True) °C', fontweight='bold', fontsize=12)
    plt.title('Distribution of Prediction Errors Across Benchmark Architectures', fontweight='bold', fontsize=14, pad=15)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
    
    plt.tight_layout()
    violin_path = os.path.join(output_dir, 'error_distribution_violins.png')
    plt.savefig(violin_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # ── Plot 4: Grouped Bar Chart of Survival Metrics (Accuracy, F1 Score, ROC AUC) ──
    plt.figure(figsize=(16, 7))
    x_b = np.arange(n_models)
    width_b = 0.28
    
    accs = [metrics_summary[m]['acc'] for m in models]
    f1s = [metrics_summary[m]['f1'] for m in models]
    rocs = [metrics_summary[m]['roc_auc'] for m in models]
    
    bars_acc = plt.bar(x_b - width_b, accs, width_b, label='Accuracy (≥60°C)', color='#10b981', alpha=0.95, edgecolor='none')
    bars_f1 = plt.bar(x_b, f1s, width_b, label='F1 Score', color='#0284c7', alpha=0.95, edgecolor='none')
    bars_roc = plt.bar(x_b + width_b, rocs, width_b, label='ROC AUC', color='#8b5cf6', alpha=0.95, edgecolor='none')
    
    plt.ylabel('Score (0.0 to 1.0)', fontweight='bold', fontsize=13)
    plt.title('Survival Classification Metrics at High-Temperature Threshold (Tm ≥ 60°C)', fontweight='bold', fontsize=15, pad=15)
    plt.xticks(x_b, models, rotation=20, ha='right', fontweight='bold', fontsize=11)
    plt.ylim(0, 1.15)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=12)
    
    for b_list in [bars_acc, bars_f1, bars_roc]:
        for bar in b_list:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold', fontsize=9, color='#1e293b')
            
    plt.tight_layout()
    survival_metrics_path = os.path.join(output_dir, 'survival_classification_metrics.png')
    plt.savefig(survival_metrics_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # ── Plot 5: Actual Continuous ROC Curves for Tm ≥ 60°C Survival Prediction ──
    plt.figure(figsize=(10, 8))
    try:
        from sklearn.metrics import roc_curve
        for i, name in enumerate(models):
            y_t = results[name]['y_true']
            y_p = results[name]['y_pred']
            y_t_bin = (y_t >= 60.0).astype(int)
            fpr, tpr, _ = roc_curve(y_t_bin, y_p)
            plt.plot(fpr, tpr, linewidth=2.5, color=colors[i % len(colors)], label=f"{name} (AUC = {metrics_summary[name]['roc_auc']:.3f})")
    except Exception:
        pass
        
    plt.plot([0, 1], [0, 1], color='#94a3b8', linestyle='--', linewidth=1.5, label='Random Chance')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate (FPR)', fontweight='bold', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontweight='bold', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves for Tm ≥ 60°C Survival', fontweight='bold', fontsize=14, pad=15)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none', fontsize=11)
    
    plt.tight_layout()
    roc_curves_path = os.path.join(output_dir, 'roc_curves_survival_60c.png')
    plt.savefig(roc_curves_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nMasterpiece publication figures stored successfully in: {output_dir}/")
    print(f"  - {os.path.basename(bar_path)}")
    print(f"  - {os.path.basename(scatter_path)}")
    print(f"  - {os.path.basename(violin_path)}")
    print(f"  - {os.path.basename(survival_metrics_path)}")
    print(f"  - {os.path.basename(roc_curves_path)}")

if __name__ == "__main__":
    main()
