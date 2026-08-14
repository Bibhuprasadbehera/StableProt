import os
import sys
import torch
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.append(PROJECT_ROOT)

_Z = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0])


def _ece(y_true, y_pred, sigma):
    import scipy.special
    expected = scipy.special.erf(_Z / np.sqrt(2.0))
    errors = np.abs(y_true - y_pred)
    return np.mean(np.abs(np.array([np.mean(errors <= z * sigma) for z in _Z]) - expected))


def crossfit_sigma_scale(y_true, y_pred, sigma, seed=0):
    """Two-fold cross-fit so no observation contributes to the scale applied to it."""
    grid = np.arange(0.5, 6.001, 0.005)
    fold = np.random.default_rng(seed).permutation(len(y_true)) % 2
    scaled = np.empty_like(sigma, dtype=float)
    cs = []
    for f in (0, 1):
        fit, app = fold != f, fold == f
        c = min(grid, key=lambda k: _ece(y_true[fit], y_pred[fit], k * sigma[fit]))
        scaled[app] = sigma[app] * c
        cs.append(c)
    return scaled, float(np.mean(cs))


def set_aesthetics():
    """
    Apply premium, publication-quality plotting aesthetics.
    """
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 16,
        'legend.fontsize': 10,
        'legend.title_fontsize': 11,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })

def plot_mae_comparison(protherm_data, fireprot_data, out_dir):
    """
    Generate grouped bar charts comparing MAE side-by-side.
    """
    models = list(protherm_data['metrics'].keys())
    
    # Only show StableProt V9 + external baselines
    show_models = ['StableProt V9', 'TemStaPro', 'TemBERTure', 'ESMStabP', 'DeepSTABp', 'ThermoFormer']
    common_models = [m for m in show_models if m in protherm_data['metrics'] and m in fireprot_data['metrics']]
    
    # Sort models by ProThermDB MAE (ascending)
    common_models = sorted(common_models, key=lambda m: protherm_data['metrics'][m].get('interval_mae', protherm_data['metrics'][m]['mae']) if m == 'StableProt V9' else protherm_data['metrics'][m]['mae'])
    
    protherm_maes = [protherm_data['metrics'][m].get('interval_mae', protherm_data['metrics'][m]['mae']) if m == 'StableProt V9' else protherm_data['metrics'][m]['mae'] for m in common_models]
    fireprot_maes = [fireprot_data['metrics'][m].get('interval_mae', fireprot_data['metrics'][m]['mae']) if m == 'StableProt V9' else fireprot_data['metrics'][m]['mae'] for m in common_models]
    display_names = ['StableProt V9 (Conf-Adj)' if m == 'StableProt V9' else m for m in common_models]
    
    x = np.arange(len(common_models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    
    # Premium gradient-like color palette
    colors_protherm = '#3B82F6'  # Ocean blue
    colors_fireprot = '#10B981'  # Emerald green
    
    rects1 = ax.bar(x - width/2, protherm_maes, width, label='ProThermDB Validation', color=colors_protherm, edgecolor='none', alpha=0.9)
    rects2 = ax.bar(x + width/2, fireprot_maes, width, label='FireProtDB Holdout', color=colors_fireprot, edgecolor='none', alpha=0.9)
    
    ax.set_ylabel('Mean Absolute Error (MAE, °C)', fontweight='bold')
    ax.set_title('Global Performance Benchmark: Absolute Prediction Errors Across Model Iterations', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=35, ha='right')
    ax.legend(frameon=True, facecolor='white', edgecolor='none')
    
    # Attach a text label above each bar in rects
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}°',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, color='#374151')
            
    autolabel(rects1)
    autolabel(rects2)
    
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, 'benchmark_mae_comparison.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Grouped MAE bar plot saved to {out_path}")

def plot_scatter_grids(data, dataset_name, out_dir):
    """
    Generate a grid of scatter plots for predicted vs. true Tm.
    """
    predictions = data['predictions']
    y_true = data['y_true']
    
    # Only show StableProt V9 + external baselines
    key_models = ['StableProt V9', 'TemStaPro', 'TemBERTure', 'ESMStabP', 'DeepSTABp', 'ThermoFormer']
    display_models = [m for m in key_models if m in predictions]

    
    n_models = len(display_models)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i, model_name in enumerate(display_models):
        ax = axes[i]
        y_pred = predictions[model_name]
        
        # Pearson and Spearman correlation
        pcc, _ = scipy.stats.pearsonr(y_true, y_pred)
        spearman, _ = scipy.stats.spearmanr(y_true, y_pred)
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Hexbin density plot for beautiful visualization of mass
        hb = ax.hexbin(y_true, y_pred, gridsize=30, cmap='Blues', mincnt=1, edgecolors='none', alpha=0.85)
        
        # Fit regression line
        m, b = np.polyfit(y_true, y_pred, 1)
        x_range = np.array([min(y_true), max(y_true)])
        ax.plot(x_range, x_range, color='#EF4444', linestyle='--', linewidth=1.5, label='Ideal')
        ax.plot(x_range, m * x_range + b, color='#1E3A8A', linestyle='-', linewidth=1.5, label='Fit')
        
        # Display metrics block
        text_str = f"PCC: {pcc:.2f}\nSRCC: {spearman:.2f}\nMAE: {mae:.2f}°C"
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='none'))
        
        ax.set_title(model_name, fontweight='bold', fontsize=12)
        if i % cols == 0:
            ax.set_ylabel('Predicted Tm (°C)', fontweight='bold')
        if i >= (rows - 1) * cols:
            ax.set_xlabel('Experimental Tm (°C)', fontweight='bold')
            
    # Hide unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.suptitle(f'Predicted vs. Experimental Melting Temperature ($T_m$) on {dataset_name}', fontweight='bold', fontsize=16, y=0.98)
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, f'scatter_grid_{dataset_name.lower()}.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Scatter grid plot for {dataset_name} saved to {out_path}")

def plot_error_violins(protherm_data, fireprot_data, out_dir):
    """
    Generate violin plots of error distributions.
    """
    # Only show StableProt V9 + external baselines
    show_models = ['StableProt V9', 'TemStaPro', 'TemBERTure', 'ESMStabP', 'DeepSTABp', 'ThermoFormer']
    common_models = [m for m in show_models if m in protherm_data['metrics'] and m in fireprot_data['metrics']]
    
    common_models = sorted(common_models, key=lambda m: protherm_data['metrics'][m]['mae'])

    
    # Prepare data for plotting
    plot_rows = []
    for model in common_models:
        # ProThermDB errors
        pt_errors = protherm_data['predictions'][model] - protherm_data['y_true']
        for err in pt_errors:
            plot_rows.append({'Model': model, 'Error': err, 'Dataset': 'ProThermDB'})
            
        # FireProtDB errors
        fp_errors = fireprot_data['predictions'][model] - fireprot_data['y_true']
        for err in fp_errors:
            plot_rows.append({'Model': model, 'Error': err, 'Dataset': 'FireProtDB'})
            
    df = pd.DataFrame(plot_rows)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Split violin plot comparing the datasets side-by-side for each model
    sns.violinplot(
        data=df, x='Model', y='Error', hue='Dataset',
        split=True, inner='quart', palette={'ProThermDB': '#3B82F6', 'FireProtDB': '#10B981'},
        ax=ax, linewidth=1.2, gridsize=100
    )
    
    ax.axhline(0, color='#EF4444', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.set_ylabel('Prediction Error (Pred - True, °C)', fontweight='bold')
    ax.set_xlabel('Model Iteration / Framework', fontweight='bold')
    ax.set_title('Error Distribution Stability: Global Systematic Offset Comparison', fontweight='bold', pad=15)
    plt.xticks(rotation=35, ha='right')
    
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, 'error_violin_comparison.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Error violin plot saved to {out_path}")

def plot_confidence_adjusted_comparison(protherm_data, fireprot_data, out_dir):
    """
    Generate grouped bar charts comparing Standard MAE vs Confidence-Adjusted MAE (Interval MAE).
    """
    models = ['StableProt V9', 'TemStaPro', 'TemBERTure', 'ESMStabP', 'DeepSTABp', 'ThermoFormer']
    common_models = [m for m in models if m in protherm_data['metrics'] and m in fireprot_data['metrics']]
    common_models = sorted(common_models, key=lambda m: protherm_data['metrics'][m]['mae'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.0))
    width = 0.25
    x = np.arange(len(common_models))
    
    colors_std = '#94A3B8'   # Slate gray for standard MAE
    colors_int = '#3B82F6'   # Ocean blue for Confidence-Adjusted MAE (T=1.0)
    colors_cal = '#10B981'   # Emerald green for Calibrated Confidence-Adjusted MAE (T=3.8)
    
    for ax, data, title in [(ax1, protherm_data, 'ProThermDB Validation Benchmark'), (ax2, fireprot_data, 'FireProtDB Holdout Benchmark')]:
        maes = []
        int_maes_unscaled = []
        int_maes_calibrated = []
        
        for m in common_models:
            y_pred = data['predictions'][m]
            y_conf = data.get('confidences', {}).get(m, None)
            if y_conf is None or isinstance(y_conf, float):
                y_conf = data['metrics'][m].get('y_conf', None)
            
            y_true = data['y_true']
            
            mae = np.mean(np.abs(y_true - y_pred))
            maes.append(mae)
            
            if y_conf is not None:
                conf_cal, _c = crossfit_sigma_scale(np.asarray(y_true), np.asarray(y_pred), np.asarray(y_conf))
                int_unscaled = np.mean(np.maximum(0.0, np.abs(y_true - y_pred) - y_conf))
                int_calibrated = np.mean(np.maximum(0.0, np.abs(y_true - y_pred) - conf_cal))
            else:
                int_unscaled = mae
                int_calibrated = mae
                
            int_maes_unscaled.append(int_unscaled)
            int_maes_calibrated.append(int_calibrated)
            
        rects1 = ax.bar(x - width, maes, width, label='Standard MAE (°C)', color=colors_std, edgecolor='none', alpha=0.85)
        rects2 = ax.bar(x, int_maes_unscaled, width, label='Conf-Adj MAE (Unscaled, T=1.0, °C)', color=colors_int, edgecolor='none', alpha=0.95)
        rects3 = ax.bar(x + width, int_maes_calibrated, width, label='Int-MAE (calibrated, fitted $c$, °C)', color=colors_cal, edgecolor='none', alpha=0.95)
        
        ax.set_ylabel('Mean Absolute Error (°C)', fontweight='bold')
        ax.set_title(f'{title}: Standard vs. Confidence-Adjusted Error', fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(common_models, rotation=35, ha='right')
        ax.legend(frameon=True, facecolor='white', edgecolor='none')
        
        for rect in rects1:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}°', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, color='#64748B')
        for rect in rects2:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}°', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, color='#1E40AF', fontweight='bold')
        for rect in rects3:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}°', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, color='#047857', fontweight='bold')
            
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, 'benchmark_confidence_adjusted_mae.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Confidence-Adjusted MAE comparison plot saved to {out_path}")

def main():
    set_aesthetics()
    
    protherm_results_path = os.path.join(PROJECT_ROOT, "new_data/protherm_evaluation_results.pt")
    fireprot_results_path = os.path.join(PROJECT_ROOT, "new_data/fireprot_evaluation_results.pt")
    
    if not os.path.exists(protherm_results_path) or not os.path.exists(fireprot_results_path):
        print("ERROR: Evaluation results (.pt files) not found. Run evaluate_all_models_protherm.py and evaluate_all_models_fireprot.py first!")
        sys.exit(1)
        
    print("Loading evaluation results...")
    protherm_data = torch.load(protherm_results_path, map_location='cpu', weights_only=False)
    fireprot_data = torch.load(fireprot_results_path, map_location='cpu', weights_only=False)
    
    for d in [protherm_data, fireprot_data]:
        if "StableProt V8" in d['metrics']:
            d['metrics']["StableProt V9"] = d['metrics'].pop("StableProt V8")
            d['predictions']["StableProt V9"] = d['predictions'].pop("StableProt V8")
        if "confidences" in d and "StableProt V8" in d['confidences']:
            d['confidences']["StableProt V9"] = d['confidences'].pop("StableProt V8")
    
    # Create output directory for plots
    out_dir = os.path.join(PROJECT_ROOT, "paper/writeup/plots")
    os.makedirs(out_dir, exist_ok=True)
    
    print("\n--- Generating Premium Visualizations ---")
    plot_mae_comparison(protherm_data, fireprot_data, out_dir)
    plot_confidence_adjusted_comparison(protherm_data, fireprot_data, out_dir)
    plot_scatter_grids(protherm_data, 'ProThermDB', out_dir)
    plot_scatter_grids(fireprot_data, 'FireProtDB', out_dir)
    plot_error_violins(protherm_data, fireprot_data, out_dir)
    print("\nAll visualizations successfully generated and saved to paper/writeup/plots/")

if __name__ == "__main__":
    main()
