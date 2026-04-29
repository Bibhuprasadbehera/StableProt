"""
Compare results across ALL experiment versions (v0, v1, v2, and any future ones).

Automatically discovers experiment directories and generates:
  - Side-by-side ROC curve comparisons per threshold
  - Bar chart comparison of key metrics
  - Summary comparison table with improvement deltas

Usage:
    python compare_results.py                              # Auto-discover all vX_* folders
    python compare_results.py --thresholds 40 65           # Specific thresholds only
    python compare_results.py --experiments v0_original v2_improved  # Specific experiments
"""

import argparse
import glob
import json
import os
import sys
import torch
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from common.metrics import (
    load_metrics, plot_comparison_roc, plot_metrics_bar_comparison,
    compute_all_metrics, print_metrics
)


# ── Display names for experiment folders ──
DISPLAY_NAMES = {
    'v0_original': 'V0 Original (pre-trained)',
    'v1_baseline': 'V1 Baseline (retrained, no reg.)',
    'v2_improved': 'V2 Improved (weighted+balanced+dropout)',
}


def discover_experiments(experiments_dir):
    """Auto-discover experiment directories that have results."""
    experiments = {}
    for entry in sorted(os.listdir(experiments_dir)):
        full_path = os.path.join(experiments_dir, entry)
        results_path = os.path.join(full_path, 'results')
        if os.path.isdir(full_path) and entry.startswith('v') and os.path.isdir(results_path):
            display_name = DISPLAY_NAMES.get(entry, entry)
            experiments[entry] = {
                'dir': results_path,
                'name': display_name,
            }
    return experiments


def load_experiment_results(results_dir, thresholds):
    """Load ensemble predictions and metrics from an experiment directory."""
    results = {}
    for t in thresholds:
        ensemble_dir = os.path.join(results_dir, "t%d" % t, "ensemble")
        metrics_path = os.path.join(ensemble_dir, 'metrics.json')
        preds_path = os.path.join(ensemble_dir, 'predictions.pt')

        if os.path.exists(metrics_path) and os.path.exists(preds_path):
            results[t] = {
                'metrics': load_metrics(metrics_path),
                'predictions': torch.load(preds_path),
            }
    return results


def main():
    parser = argparse.ArgumentParser(description='Compare all experiment versions')
    parser.add_argument('--thresholds', type=int, nargs='+',
                        default=[40, 45, 50, 55, 60, 65],
                        help='Thresholds to compare (default: 40 45 50 55 60 65)')
    parser.add_argument('--experiments', type=str, nargs='+', default=None,
                        help='Specific experiment folder names (default: auto-discover all)')
    args = parser.parse_args()

    output_dir = os.path.join(SCRIPT_DIR, 'comparison')
    os.makedirs(output_dir, exist_ok=True)

    # ── Discover or select experiments ──
    all_experiments = discover_experiments(SCRIPT_DIR)

    if args.experiments:
        # Filter to requested experiments
        selected = {}
        for name in args.experiments:
            if name in all_experiments:
                selected[name] = all_experiments[name]
            else:
                print("WARNING: Experiment '%s' not found or has no results/ directory." % name)
                print("  Available: %s" % list(all_experiments.keys()))
        experiments = selected
    else:
        experiments = all_experiments

    if len(experiments) < 2:
        print("ERROR: Need at least 2 experiments to compare.")
        print("  Found: %s" % list(experiments.keys()))
        print("\nMake sure you have run at least 2 experiments:")
        print("  cd v0_original && python evaluate.py")
        print("  cd v1_baseline && python train.py")
        print("  cd v2_improved && python train.py")
        sys.exit(1)

    print("=" * 75)
    print("  EXPERIMENT COMPARISON")
    print("=" * 75)
    print("  Experiments found:")
    for key, info in experiments.items():
        print("    • %-20s → %s" % (key, info['name']))
    print("  Thresholds: %s" % args.thresholds)
    print("=" * 75)

    # ── Load all results ──
    loaded = {}
    for key, info in experiments.items():
        print("\nLoading %s from: %s" % (key, info['dir']))
        results = load_experiment_results(info['dir'], args.thresholds)
        if results:
            loaded[key] = {'results': results, 'name': info['name']}
            print("  → Loaded %d threshold(s): %s" % (
                len(results), sorted(results.keys())))
        else:
            print("  → WARNING: No results found!")

    if len(loaded) < 2:
        print("\nERROR: Only %d experiment(s) have results. Need ≥ 2." % len(loaded))
        sys.exit(1)

    # ── Per-threshold comparison ──
    comparison = {}
    exp_keys = list(loaded.keys())  # Ordered list of experiment names
    exp_names = [loaded[k]['name'] for k in exp_keys]

    for t in args.thresholds:
        # Check which experiments have results for this threshold
        available = [k for k in exp_keys if t in loaded[k]['results']]
        if len(available) < 2:
            print("\nSkipping threshold %d°C — only %d experiment(s) have results." % (
                t, len(available)))
            continue

        print("\n" + "─" * 75)
        print("  Threshold: %d°C" % t)
        print("─" * 75)

        # Build header
        header = "  %-22s" % "Metric"
        for k in available:
            header += " %-20s" % loaded[k]['name'][:20]
        print(header)
        print("  " + "-" * (22 + 20 * len(available)))

        # Print each metric
        metric_keys = ['auc_roc', 'auc_prc', 'f1', 'mcc', 'balanced_accuracy',
                        'sensitivity', 'specificity', 'precision']

        threshold_data = {}
        for k in available:
            threshold_data[k] = loaded[k]['results'][t]['metrics']

        for metric_key in metric_keys:
            row = "  %-22s" % metric_key
            values = []
            for k in available:
                val = threshold_data[k].get(metric_key, 0)
                values.append(val)
                row += " %-20.4f" % val
            # Add delta between last and first
            if len(values) >= 2:
                delta = values[-1] - values[0]
                arrow = "↑" if delta > 0.001 else ("↓" if delta < -0.001 else "=")
                row += "  Δ=%+.4f %s" % (delta, arrow)
            print(row)

        comparison["t%d" % t] = threshold_data

        # ── ROC curve comparison for this threshold ──
        roc_data = {}
        for k in available:
            preds = loaded[k]['results'][t]['predictions']
            roc_data[loaded[k]['name']] = (preds['y_true'], preds['y_prob'])

        plot_comparison_roc(
            roc_data,
            title="ROC Comparison — Threshold %d°C" % t,
            save_path=os.path.join(output_dir, 'roc_comparison_t%d.png' % t)
        )

        # ── Bar chart comparison for this threshold ──
        metrics_list = [threshold_data[k] for k in available]
        labels_list = [loaded[k]['name'][:25] for k in available]
        plot_metrics_bar_comparison(
            metrics_list, labels_list,
            save_path=os.path.join(output_dir, 'metrics_comparison_t%d.png' % t)
        )

    # ── Grand summary table ──
    print("\n\n" + "=" * 90)
    print("  GRAND SUMMARY — AUC-ROC across all thresholds")
    print("=" * 90)

    # Header
    header = "  %-12s" % "Threshold"
    for k in exp_keys:
        header += " %-22s" % loaded[k]['name'][:22]
    print(header)
    print("  " + "-" * (12 + 22 * len(exp_keys)))

    for t in args.thresholds:
        row = "  %-12s" % ("%d°C" % t)
        values = []
        for k in exp_keys:
            if t in loaded[k]['results']:
                val = loaded[k]['results'][t]['metrics']['auc_roc']
                values.append(val)
                row += " %-22.4f" % val
            else:
                values.append(None)
                row += " %-22s" % "N/A"

        # Best marker
        valid_values = [v for v in values if v is not None]
        if valid_values:
            best = max(valid_values)
            row += "  best=%.4f" % best
        print(row)

    print("=" * 90)

    # ── MCC summary table ──
    print("\n" + "=" * 90)
    print("  GRAND SUMMARY — MCC across all thresholds")
    print("=" * 90)

    header = "  %-12s" % "Threshold"
    for k in exp_keys:
        header += " %-22s" % loaded[k]['name'][:22]
    print(header)
    print("  " + "-" * (12 + 22 * len(exp_keys)))

    for t in args.thresholds:
        row = "  %-12s" % ("%d°C" % t)
        for k in exp_keys:
            if t in loaded[k]['results']:
                val = loaded[k]['results'][t]['metrics']['mcc']
                row += " %-22.4f" % val
            else:
                row += " %-22s" % "N/A"
        print(row)

    print("=" * 90)

    # ── Save comparison JSON ──
    save_path = os.path.join(output_dir, 'comparison.json')
    with open(save_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print("\nComparison data saved to: %s" % save_path)
    print("Plots saved to: %s/" % output_dir)
    print("\nPlots generated:")
    for f_name in sorted(os.listdir(output_dir)):
        if f_name.endswith('.png'):
            print("  📊 %s" % f_name)


if __name__ == '__main__':
    main()
