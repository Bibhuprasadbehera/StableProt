"""
V0 Original — Evaluate pre-trained TemStaPro models (no retraining).

Loads the existing models from StableProt/models/ and evaluates them on
the prepared test data. This provides the true baseline to compare against.

The original models were trained by the TemStaPro authors using PyTorch Lightning.

Usage:
    python evaluate.py                          # Evaluate all thresholds
    python evaluate.py --thresholds 40 65       # Evaluate specific thresholds
    python evaluate.py --data ../prepared_data.pt
"""

import argparse
import json
import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Add experiments root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(EXPERIMENTS_DIR)
STABLEPROT_DIR = os.path.join(PROJECT_ROOT, "StableProt")

sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, STABLEPROT_DIR)

from utils.data_utils import prepare_data_for_threshold
from utils.metrics import (
    compute_all_metrics, print_metrics, save_metrics,
    plot_roc_curve, plot_prc_curve
)

# Import the ORIGINAL MLP_C2H2 model from StableProt
from MLP import MLP_C2H2


def load_original_model(model_path, device='cpu'):
    """
    Load a pre-trained TemStaPro model.

    The original models were saved via PyTorch Lightning, so state_dict keys
    have a 'model.model.' prefix that needs to be remapped to 'model.'.
    Auto-detects hidden layer sizes from checkpoint weights.
    """
    state_dict = torch.load(model_path, map_location=torch.device(device))['state_dict']

    # Remap keys: 'model.model.X' -> 'model.X'
    new_state_dict = {}
    for key in list(state_dict.keys()):
        new_key = key.replace('model.model.', 'model.')
        new_state_dict[new_key] = state_dict[key]

    # Auto-detect hidden layer sizes from weight shapes
    # Layer 0: Linear(input, hidden1) -> weight shape is (hidden1, input)
    # Layer 2: Linear(hidden1, hidden2) -> weight shape is (hidden2, hidden1)
    h1 = new_state_dict['model.0.weight'].shape[0]  # hidden_size_1
    h2 = new_state_dict['model.2.weight'].shape[0]  # hidden_size_2
    input_size = new_state_dict['model.0.weight'].shape[1]

    classifier = MLP_C2H2(input_size=input_size, hidden_size_1=h1, hidden_size_2=h2)
    classifier.load_state_dict(new_state_dict)
    classifier.eval()
    classifier.to(device)
    print("    Architecture: %d → %d → %d → 1" % (input_size, h1, h2))
    return classifier


def run_evaluation(args):
    """Run evaluation of pre-trained models."""

    # ── Setup ──
    output_dir = os.path.join(SCRIPT_DIR, 'results')
    os.makedirs(output_dir, exist_ok=True)

    data_path = args.data if args.data else os.path.join(EXPERIMENTS_DIR, 'prepared_data_full.pt')
    models_dir = os.path.join(STABLEPROT_DIR, 'models')

    if not os.path.exists(data_path):
        print("ERROR: Prepared data not found: %s" % data_path)
        print("Run `python prepare_data.py` first!")
        sys.exit(1)

    if not os.path.exists(models_dir):
        print("ERROR: Pre-trained models not found: %s" % models_dir)
        sys.exit(1)

    thresholds = args.thresholds if args.thresholds else [40, 45, 50, 55, 60, 65]
    seeds = [1, 2, 3, 4, 5]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("  V0 ORIGINAL — Pre-trained TemStaPro Models")
    print("=" * 60)
    print("  Models dir: %s" % models_dir)
    print("  Thresholds: %s" % thresholds)
    print("  Seeds: %s" % seeds)
    print("  Device: %s" % device)
    print("  NOTE: No training — evaluation only!")
    print("=" * 60)

    # List available models
    available_models = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
    print("\n  Available pre-trained models: %d" % len(available_models))

    all_results = {}

    for threshold in thresholds:
        print("\n" + "-" * 50)
        print("  Threshold: %d°C" % threshold)
        print("-" * 50)

        # Load data for this threshold
        data = prepare_data_for_threshold(data_path, threshold)

        # Check which seed models exist for this threshold
        threshold_predictions = []
        found_seeds = []

        for seed in seeds:
            model_filename = "mean_major_imbal-%d_s%d.pt" % (threshold, seed)
            model_path = os.path.join(models_dir, model_filename)

            if not os.path.exists(model_path):
                print("  WARNING: Model not found: %s" % model_filename)
                continue

            print("\n  Loading: %s" % model_filename)
            model = load_original_model(model_path, device=device)

            # Run inference on test data
            model.eval()
            with torch.no_grad():
                test_input = data['test_emb'].float().to(device)
                # Original model uses ModuleList with Sigmoid, outputs probabilities
                outputs = test_input
                for layer in model.model:
                    outputs = layer(outputs)
                test_probs = outputs.squeeze().cpu()

            metrics = compute_all_metrics(data['test_labels'], test_probs)
            print_metrics(metrics, "Test Results (Seed %d, t=%d°C)" % (seed, threshold))

            # Save per-seed results
            seed_dir = os.path.join(output_dir, "t%d" % threshold, "seed%d" % seed)
            os.makedirs(seed_dir, exist_ok=True)

            save_metrics(metrics, os.path.join(seed_dir, 'metrics.json'))
            torch.save({
                'y_true': data['test_labels'],
                'y_prob': test_probs,
            }, os.path.join(seed_dir, 'predictions.pt'))

            plot_roc_curve(
                data['test_labels'], test_probs,
                title="ROC — Original t=%d°C (seed=%d)" % (threshold, seed),
                save_path=os.path.join(seed_dir, 'roc_curve.png')
            )

            threshold_predictions.append(test_probs)
            found_seeds.append(seed)

        if not threshold_predictions:
            print("  ERROR: No models found for threshold %d°C" % threshold)
            continue

        # ── Ensemble results (average across seeds) ──
        ensemble_probs = torch.stack(threshold_predictions).mean(dim=0)
        ensemble_metrics = compute_all_metrics(data['test_labels'], ensemble_probs)
        print_metrics(ensemble_metrics,
                      "ENSEMBLE Results (t=%d°C, %d seeds)" % (threshold, len(found_seeds)))

        ensemble_dir = os.path.join(output_dir, "t%d" % threshold, "ensemble")
        os.makedirs(ensemble_dir, exist_ok=True)
        save_metrics(ensemble_metrics, os.path.join(ensemble_dir, 'metrics.json'))
        torch.save({
            'y_true': data['test_labels'],
            'y_prob': ensemble_probs,
        }, os.path.join(ensemble_dir, 'predictions.pt'))

        plot_roc_curve(
            data['test_labels'], ensemble_probs,
            title="ROC — Original Ensemble t=%d°C" % threshold,
            save_path=os.path.join(ensemble_dir, 'roc_curve.png')
        )
        plot_prc_curve(
            data['test_labels'], ensemble_probs,
            title="PRC — Original Ensemble t=%d°C" % threshold,
            save_path=os.path.join(ensemble_dir, 'prc_curve.png')
        )

        all_results["t%d" % threshold] = ensemble_metrics

    # ── Save summary ──
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\n\nSummary saved to: %s" % summary_path)

    # Print final summary table
    print("\n" + "=" * 70)
    print("  V0 ORIGINAL — FINAL SUMMARY (Ensemble)")
    print("=" * 70)
    print("  %-10s %-10s %-10s %-10s %-10s %-10s" % (
        'Threshold', 'AUC-ROC', 'AUC-PRC', 'F1', 'MCC', 'Bal.Acc'))
    print("  " + "-" * 60)
    for t in thresholds:
        key = "t%d" % t
        if key in all_results:
            m = all_results[key]
            print("  %-10s %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f" % (
                "%d°C" % t, m['auc_roc'], m['auc_prc'], m['f1'], m['mcc'],
                m['balanced_accuracy']))
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='V0 Original — Evaluate pre-trained TemStaPro models')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to prepared_data.pt')
    parser.add_argument('--thresholds', type=int, nargs='+', default=None,
                        help='Temperature thresholds (default: 40-65)')
    args = parser.parse_args()

    run_evaluation(args)


if __name__ == '__main__':
    main()
