"""
Compare results across ALL experiment versions (Binary and Regression).

Converts binary classifications across thresholds into continuous OGT estimates,
allowing direct comparison with regression models.

Generates:
  - Grand unified comparison table (MAE, RMSE, Spearman)
  - Binned performance (Psychro, Meso, Thermo, Hyperthermo)
  - Scatter plots (True OGT vs Predicted OGT)
"""

import argparse
import os
import sys
import torch
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# ── Display names for experiment folders ──
DISPLAY_NAMES = {
    'v0_original': 'V0 Original (pre-trained)',
    'v1_baseline': 'V1 Baseline (0-100 binary)',
    'v2_improved': 'V2 Improved (0-100 binary)',
    'v3_regression': 'V3 Regression (continuous)',
}

def get_temperature_bins():
    bins = []
    for i in range(0, 100, 10):
        start = i
        end = i + 10
        if i == 0:
            name = f"0-10°C"
            bins.append((name, lambda t, s=start, e=end: t < e))
        elif i == 90:
            name = f"90-100°C"
            bins.append((name, lambda t, s=start: t >= s))
        else:
            name = f"{start}-{end}°C"
            bins.append((name, lambda t, s=start, e=end: (t >= s) and (t < e)))
    return bins

def discover_experiments(experiments_dir):
    """Auto-discover experiment directories that have results."""
    experiments = {}
    for entry in sorted(os.listdir(experiments_dir)):
        full_path = os.path.join(experiments_dir, entry)
        results_path = os.path.join(full_path, 'results')
        if os.path.isdir(full_path) and entry.startswith('v') and os.path.isdir(results_path):
            display_name = DISPLAY_NAMES.get(entry, entry)
            # Determine if it's regression or binary
            is_regression = os.path.exists(os.path.join(results_path, 'ensemble', 'predictions.pt'))
            experiments[entry] = {
                'dir': results_path,
                'name': display_name,
                'is_regression': is_regression
            }
    return experiments

def load_and_convert_predictions(exp_info):
    """
    Loads predictions.
    If binary: converts probabilities across thresholds to a single expected OGT.
    E[T] = \int P(T > t) dt \approx \sum P(T > t_i) * \Delta t
    """
    results_dir = exp_info['dir']
    
    if exp_info['is_regression']:
        preds_path = os.path.join(results_dir, 'ensemble', 'predictions.pt')
        if not os.path.exists(preds_path):
            return None, None
        data = torch.load(preds_path)
        return data['y_true'].numpy(), data['y_pred'].numpy()
    
    else:
        # Binary experiment
        thresholds = []
        for t_dir in os.listdir(results_dir):
            if t_dir.startswith('t') and t_dir[1:].isdigit():
                thresholds.append(int(t_dir[1:]))
        thresholds.sort()
        
        if not thresholds:
            return None, None
            
        # We need the true temperatures, we can get them from the original data
        # But wait, binary predictions only saved y_true (binary labels) and y_prob.
        # We need the actual test temperatures. 
        # Let's load the full prepared data to get true test temps.
        try:
            data = torch.load(os.path.join(SCRIPT_DIR, 'prepared_data_full.pt'))
            y_true_temp = np.array(data['test_temps'])
        except:
            try:
                data = torch.load(os.path.join(SCRIPT_DIR, 'prepared_data.pt'))
                y_true_temp = np.array(data['test_temps'])
            except:
                print("ERROR: Cannot load prepared_data.pt to get true temperatures.")
                return None, None

        # Build matrix of probabilities: (num_samples, num_thresholds)
        n_samples = len(y_true_temp)
        prob_matrix = np.zeros((n_samples, len(thresholds)))
        
        valid_thresholds = []
        for i, t in enumerate(thresholds):
            preds_path = os.path.join(results_dir, f't{t}', 'ensemble', 'predictions.pt')
            if os.path.exists(preds_path):
                pred_data = torch.load(preds_path)
                probs = pred_data['y_prob'].numpy()
                if len(probs) == n_samples:
                    prob_matrix[:, i] = probs
                    valid_thresholds.append(t)
        
        if not valid_thresholds:
            return None, None
            
        # Convert to OGT. 
        # E[T] = Base_Temp + sum(P(T > t) * step)
        # We assume base temp is around the first threshold (e.g. 5 or 40)
        # A simple approximation: just use the sum of probabilities * step size, plus offset.
        # If thresholds are 40, 45, 50... step=5. Base=35? 
        # To be robust, let's just do numerical integration of the survival function P(T>t)
        # T = \int_{0}^{\infty} P(T>t) dt. 
        # Let's assume P(T>0) = 1 up to the first threshold.
        
        step_sizes = np.diff(valid_thresholds)
        # Assume uniform step size for the ends
        step = step_sizes[0] if len(step_sizes) > 0 else 5
        
        base_temp = max(0, valid_thresholds[0] - step)
        
        # Expected value
        y_pred_temp = np.full(n_samples, base_temp, dtype=float)
        for i, t in enumerate(valid_thresholds):
            # For each threshold, add prob * step to the expected value
            # This works precisely if probabilities are monotonically decreasing
            y_pred_temp += prob_matrix[:, i] * step
            
        return y_true_temp, y_pred_temp

def compute_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    spearman, _ = scipy.stats.spearmanr(y_true, y_pred)
    return {'mae': mae, 'rmse': rmse, 'spearman': spearman}

def main():
    print("=" * 80)
    print("  UNIFIED EXPERIMENT COMPARISON (OGT Metric)")
    print("=" * 80)
    
    experiments = discover_experiments(SCRIPT_DIR)
    
    results = {}
    for key, info in experiments.items():
        y_true, y_pred = load_and_convert_predictions(info)
        if y_true is not None:
            results[key] = {
                'name': info['name'],
                'y_true': y_true,
                'y_pred': y_pred
            }
            
    if not results:
        print("No results found to compare.")
        return
        
    print("\nOVERALL PERFORMANCE:")
    print("-" * 80)
    print(f"{'Experiment':<35} | {'MAE (°C)':<10} | {'RMSE (°C)':<10} | {'Spearman ρ':<10}")
    print("-" * 80)
    
    overall_metrics = {}
    for key, data in results.items():
        m = compute_metrics(data['y_true'], data['y_pred'])
        overall_metrics[key] = m
        print(f"{data['name']:<35} | {m['mae']:<10.2f} | {m['rmse']:<10.2f} | {m['spearman']:<10.3f}")
        
    print("\n\nBINNED MAE PERFORMANCE (How models perform at extremes):")
    print("-" * 105)
    
    bins = get_temperature_bins()
    
    header = f"{'Experiment':<30}"
    for name, _ in bins:
        # Abbreviate header
        abbr = name.split(' ')[0]
        header += f" | {abbr:<12}"
    print(header)
    print("-" * 105)
    
    for key, data in results.items():
        y_t = data['y_true']
        y_p = data['y_pred']
        
        row = f"{data['name'][:30]:<30}"
        
        for _, condition in bins:
            mask = np.array([condition(t) for t in y_t])
            if np.sum(mask) > 0:
                bin_mae = np.mean(np.abs(y_t[mask] - y_p[mask]))
                row += f" | {bin_mae:<12.1f}"
            else:
                row += f" | {'N/A':<12}"
        print(row)
        
    # ==========================================
    # GENERATE PUBLICATION-GRADE PLOTS
    # ==========================================
    output_dir = os.path.join(SCRIPT_DIR, 'comparison')
    os.makedirs(output_dir, exist_ok=True)
    import matplotlib.colors as mcolors
    import seaborn as sns

    # Set publication style
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        pass # Fallback to default if not found
    sns.set_context("paper", font_scale=1.2)
    
    # 1. Hexbin Scatter Plots (solves overplotting of 200k points)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (key, data) in enumerate(results.items()):
        ax = axes[i]
        hb = ax.hexbin(data['y_true'], data['y_pred'], gridsize=50, cmap='viridis', 
                       mincnt=1, norm=mcolors.LogNorm())
        ax.plot([0, 100], [0, 100], 'r--', linewidth=2, alpha=0.8)
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
        
        # Calculate R-squared and MAE for title
        mae = overall_metrics[key]['mae']
        spearman = overall_metrics[key]['spearman']
        
        ax.set_title(f"{data['name']}\nMAE: {mae:.2f}°C | Spearman ρ: {spearman:.2f}", fontweight='bold')
        ax.set_xlabel('True OGT (°C)', fontweight='bold')
        ax.set_ylabel('Predicted OGT (°C)', fontweight='bold')
        
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('Count (log scale)')

    plt.tight_layout()
    scatter_path = os.path.join(output_dir, 'scatter_hexbin_comparison.png')
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Binned MAE Grouped Bar Chart
    plt.figure(figsize=(14, 6))
    
    bin_labels = [name.split(' ')[0] for name, _ in bins]
    n_bins = len(bins)
    n_models = len(results)
    
    bar_width = 0.8 / n_models
    x = np.arange(n_bins)
    
    colors = ['#4f46e5', '#10b981', '#f59e0b', '#ec4899']
    
    for i, (key, data) in enumerate(results.items()):
        y_t = data['y_true']
        y_p = data['y_pred']
        
        bin_maes = []
        for _, condition in bins:
            mask = np.array([condition(t) for t in y_t])
            if np.sum(mask) > 0:
                bin_maes.append(np.mean(np.abs(y_t[mask] - y_p[mask])))
            else:
                bin_maes.append(0)
                
        plt.bar(x + i*bar_width - (0.8/2) + bar_width/2, bin_maes, 
                width=bar_width, label=data['name'], color=colors[i % len(colors)], alpha=0.85)
                
    plt.xticks(x, bin_labels, rotation=45, ha='right', fontweight='bold')
    plt.xlabel('True Temperature Range', fontweight='bold')
    plt.ylabel('Mean Absolute Error (°C)', fontweight='bold')
    plt.title('Error Profile Across Temperature Bins', fontweight='bold', fontsize=14)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    bar_path = os.path.join(output_dir, 'binned_mae_barplot.png')
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Error Distribution Violin Plot
    plt.figure(figsize=(10, 6))
    
    plot_data = []
    labels = []
    
    for key, data in results.items():
        errors = data['y_pred'] - data['y_true']
        plot_data.append(errors)
        labels.append(data['name'].split(' ')[0]) # Short name
        
    parts = plt.violinplot(plot_data, showmeans=True, showmedians=False, showextrema=True)
    
    # Color violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_edgecolor('black')
        pc.set_alpha(0.6)
        
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
        vp = parts[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(1)
        
    plt.axhline(0, color='red', linestyle='--', alpha=0.7, label='Zero Error')
    plt.xticks(np.arange(1, len(labels) + 1), labels, fontweight='bold')
    plt.ylabel('Prediction Error (Pred - True) °C', fontweight='bold')
    plt.title('Distribution of Prediction Errors', fontweight='bold', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    violin_path = os.path.join(output_dir, 'error_distribution_violin.png')
    plt.savefig(violin_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nPublication-grade plots saved to {output_dir}/")
    print(f"  - {os.path.basename(scatter_path)}")
    print(f"  - {os.path.basename(bar_path)}")
    print(f"  - {os.path.basename(violin_path)}")

if __name__ == '__main__':
    main()
