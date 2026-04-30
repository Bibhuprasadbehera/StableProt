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
    return [
        ('Psychrophiles (<20°C)', lambda t: t < 20),
        ('Mesophiles (20-40°C)', lambda t: (t >= 20) and (t < 40)),
        ('Thermophiles (40-60°C)', lambda t: (t >= 40) and (t < 60)),
        ('Extreme (60-80°C)', lambda t: (t >= 60) and (t < 80)),
        ('Hyperthermo (>=80°C)', lambda t: t >= 80),
    ]

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
        
    # Generate scatter plots
    output_dir = os.path.join(SCRIPT_DIR, 'comparison')
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    for i, (key, data) in enumerate(results.items()):
        plt.subplot(2, 2, i+1)
        plt.scatter(data['y_true'], data['y_pred'], alpha=0.1, s=5)
        plt.plot([0, 100], [0, 100], 'r--')
        plt.xlim(0, 105)
        plt.ylim(0, 105)
        plt.title(f"{data['name']}\nMAE: {overall_metrics[key]['mae']:.1f}°C")
        plt.xlabel('True OGT (°C)')
        plt.ylabel('Predicted OGT (°C)')
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_comparison.png'))
    print(f"\nScatter plots saved to {output_dir}/scatter_comparison.png")

if __name__ == '__main__':
    main()
