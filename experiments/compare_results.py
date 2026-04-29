"""
Compare results between V1 Baseline and V2 Improved experiments.

Loads results from both experiment directories and generates:
  - Side-by-side ROC curve comparisons per threshold
  - Bar chart comparison of key metrics
  - Summary comparison table

Usage:
    python compare_results.py
    python compare_results.py --thresholds 40 65
"""

import argparse
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
        else:
            print("  WARNING: Missing results for threshold %d°C in %s" % (t, results_dir))
    return results


def main():
    parser = argparse.ArgumentParser(description='Compare V1 vs V2 experiments')
    parser.add_argument('--thresholds', type=int, nargs='+',
                        default=[40, 45, 50, 55, 60, 65],
                        help='Thresholds to compare')
    parser.add_argument('--v1-dir', type=str,
                        default=os.path.join(SCRIPT_DIR, 'v1_baseline', 'results'),
                        help='Path to V1 results directory')
    parser.add_argument('--v2-dir', type=str,
                        default=os.path.join(SCRIPT_DIR, 'v2_improved', 'results'),
                        help='Path to V2 results directory')
    args = parser.parse_args()

    output_dir = os.path.join(SCRIPT_DIR, 'comparison')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  EXPERIMENT COMPARISON — V1 Baseline vs V2 Improved")
    print("=" * 70)

    # Load results
    print("\nLoading V1 Baseline results from: %s" % args.v1_dir)
    v1_results = load_experiment_results(args.v1_dir, args.thresholds)

    print("Loading V2 Improved results from: %s" % args.v2_dir)
    v2_results = load_experiment_results(args.v2_dir, args.thresholds)

    if not v1_results or not v2_results:
        print("\nERROR: No results found. Make sure you've run both experiments first:")
        print("  cd v1_baseline && python train.py")
        print("  cd v2_improved && python train.py")
        sys.exit(1)

    # ── Per-threshold comparison ──
    comparison = {}

    for t in args.thresholds:
        if t not in v1_results or t not in v2_results:
            continue

        print("\n" + "─" * 60)
        print("  Threshold: %d°C" % t)
        print("─" * 60)

        v1_m = v1_results[t]['metrics']
        v2_m = v2_results[t]['metrics']

        # Print side by side
        print("  %-20s %-15s %-15s %-10s" % ('Metric', 'V1 Baseline', 'V2 Improved', 'Delta'))
        print("  " + "-" * 55)

        metric_keys = ['auc_roc', 'auc_prc', 'f1', 'mcc', 'balanced_accuracy',
                        'sensitivity', 'specificity', 'precision']
        for key in metric_keys:
            v1_val = v1_m.get(key, 0)
            v2_val = v2_m.get(key, 0)
            delta = v2_val - v1_val
            arrow = "↑" if delta > 0.001 else ("↓" if delta < -0.001 else "=")
            print("  %-20s %-15.4f %-15.4f %s%.4f %s" % (
                key, v1_val, v2_val, '+' if delta >= 0 else '', delta, arrow))

        comparison["t%d" % t] = {
            'v1': v1_m,
            'v2': v2_m,
        }

        # ── ROC curve comparison for this threshold ──
        v1_preds = v1_results[t]['predictions']
        v2_preds = v2_results[t]['predictions']

        roc_data = {
            'V1 Baseline': (v1_preds['y_true'], v1_preds['y_prob']),
            'V2 Improved': (v2_preds['y_true'], v2_preds['y_prob']),
        }
        plot_comparison_roc(
            roc_data,
            title="ROC Comparison — Threshold %d°C" % t,
            save_path=os.path.join(output_dir, 'roc_comparison_t%d.png' % t)
        )

        # ── Bar chart comparison for this threshold ──
        plot_metrics_bar_comparison(
            [v1_m, v2_m],
            ['V1 Baseline', 'V2 Improved'],
            save_path=os.path.join(output_dir, 'metrics_comparison_t%d.png' % t)
        )

    # ── Summary table across all thresholds ──
    print("\n\n" + "=" * 80)
    print("  SUMMARY — V1 Baseline vs V2 Improved (Ensemble AUC-ROC)")
    print("=" * 80)
    print("  %-10s %-18s %-18s %-12s" % ('Threshold', 'V1 AUC-ROC', 'V2 AUC-ROC', 'Improvement'))
    print("  " + "-" * 58)

    for t in args.thresholds:
        key = "t%d" % t
        if key in comparison:
            v1_auc = comparison[key]['v1']['auc_roc']
            v2_auc = comparison[key]['v2']['auc_roc']
            delta = v2_auc - v1_auc
            arrow = "✓" if delta > 0 else "✗"
            print("  %-10s %-18.4f %-18.4f %+.4f %s" % (
                "%d°C" % t, v1_auc, v2_auc, delta, arrow))

    print("=" * 80)

    # ── Save comparison JSON ──
    save_path = os.path.join(output_dir, 'comparison.json')
    with open(save_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print("\nComparison saved to: %s" % save_path)
    print("Plots saved to: %s/" % output_dir)


if __name__ == '__main__':
    main()
