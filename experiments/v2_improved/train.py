"""
V2 Improved Training Script.

Key improvements over V1 Baseline:
  1. BCEWithLogitsLoss with pos_weight (upweights minority thermophilic class)
  2. WeightedRandomSampler (balanced batches during training)
  3. Dropout + BatchNorm (regularization)

Usage:
    python train.py                          # Train all thresholds
    python train.py --thresholds 40 65       # Train specific thresholds
    python train.py --data ../prepared_data.pt --epochs 30
"""

import argparse
import json
import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np

# Add experiments root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, EXPERIMENTS_DIR)

from common.data_utils import prepare_data_for_threshold, create_data_loaders
from common.metrics import (
    compute_all_metrics, print_metrics, save_metrics,
    plot_roc_curve, plot_prc_curve, plot_training_history
)
from model import MLP_Improved
from config import CONFIG


def compute_pos_weight(labels, cap=50.0):
    """
    Compute pos_weight for BCEWithLogitsLoss.
    pos_weight = n_negative / n_positive

    This tells the loss function: "a positive sample is worth pos_weight
    times more than a negative sample."
    """
    n_pos = labels.sum().item()
    n_neg = len(labels) - n_pos

    if n_pos == 0:
        return torch.tensor([cap])

    weight = n_neg / n_pos
    weight = min(weight, cap)
    return torch.tensor([weight])


def train_one_model(model, train_loader, val_loader, val_emb, val_labels,
                    criterion, optimizer, num_epochs, patience, device='cpu'):
    """
    Train a single model with early stopping.
    Uses BCEWithLogitsLoss — model outputs raw logits, not probabilities.

    Returns:
        (best_model_state, history_dict)
    """
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc_roc': [],
        'val_mcc': [],
    }

    best_val_loss = float('inf')
    best_state = None
    epochs_without_improvement = 0

    # Create a validation criterion without pos_weight for fair comparison
    val_criterion = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        # ── Training ──
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            optimizer.zero_grad()
            logits = model(batch_x).squeeze()
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_train_loss = running_loss / max(n_batches, 1)

        # ── Validation ──
        model.eval()
        with torch.no_grad():
            val_logits = model(val_emb.float().to(device)).squeeze().cpu()
            val_loss = val_criterion(val_logits, val_labels).item()

            # Convert logits to probabilities for metrics
            val_probs = torch.sigmoid(val_logits).numpy()
            val_true = val_labels.numpy()

            try:
                from sklearn.metrics import roc_auc_score, matthews_corrcoef
                val_auc = roc_auc_score(val_true, val_probs)
                val_preds = (val_probs >= 0.5).astype(float)
                val_mcc = matthews_corrcoef(val_true, val_preds)
            except Exception:
                val_auc = 0.0
                val_mcc = 0.0

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc_roc'].append(val_auc)
        history['val_mcc'].append(val_mcc)

        # ── Early stopping (on unweighted val loss) ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print("    Epoch %3d/%d — train_loss=%.4f, val_loss=%.4f, val_AUC=%.4f, val_MCC=%.4f" % (
                epoch + 1, num_epochs, avg_train_loss, val_loss, val_auc, val_mcc))

        if epochs_without_improvement >= patience:
            print("    Early stopping at epoch %d (patience=%d)" % (epoch + 1, patience))
            break

    return best_state, history


def run_experiment(args):
    """Run the full improved experiment."""

    # ── Setup ──
    output_dir = os.path.join(SCRIPT_DIR, 'results')
    os.makedirs(output_dir, exist_ok=True)

    data_path = args.data if args.data else os.path.join(EXPERIMENTS_DIR, CONFIG['data_path'])
    if not os.path.exists(data_path):
        print("ERROR: Prepared data not found: %s" % data_path)
        print("Run `python prepare_data.py` first!")
        sys.exit(1)

    thresholds = args.thresholds if args.thresholds else CONFIG['thresholds']
    seeds = CONFIG['seeds']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("  V2 IMPROVED — Weighted Loss + Balanced Sampling + Dropout")
    print("=" * 60)
    print("  Thresholds: %s" % thresholds)
    print("  Seeds: %s" % seeds)
    print("  Device: %s" % device)
    print("  Epochs: %d (patience: %d)" % (CONFIG['num_epochs'], CONFIG['early_stopping_patience']))
    print("  Learning rate: %s" % CONFIG['learning_rate'])
    print("  Batch size: %d" % CONFIG['batch_size'])
    print("  Loss: BCEWithLogitsLoss (auto pos_weight, cap=%.0f)" % CONFIG['pos_weight_cap'])
    print("  Balanced sampling: Yes (WeightedRandomSampler)")
    print("  Dropout: %.1f / %.1f" % (CONFIG['dropout_1'], CONFIG['dropout_2']))
    print("  Weight decay: %s" % CONFIG['weight_decay'])
    print("=" * 60)

    all_results = {}

    for threshold in thresholds:
        print("\n" + "─" * 50)
        print("  Threshold: %d°C" % threshold)
        print("─" * 50)

        # Load data for this threshold
        data = prepare_data_for_threshold(data_path, threshold)

        # Compute pos_weight for this threshold
        pos_weight = compute_pos_weight(data['train_labels'], cap=CONFIG['pos_weight_cap'])
        print("  pos_weight: %.2f" % pos_weight.item())

        # Create balanced data loaders
        train_loader, val_loader = create_data_loaders(
            data['train_emb'], data['train_labels'],
            data['val_emb'], data['val_labels'],
            batch_size=CONFIG['batch_size'],
            balanced=True  # ← Key difference: balanced sampling
        )

        threshold_predictions = []
        threshold_results = []

        for seed in seeds:
            print("\n  Seed %d:" % seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Create improved model
            model = MLP_Improved(
                input_size=CONFIG['input_size'],
                hidden_size_1=CONFIG['hidden_size_1'],
                hidden_size_2=CONFIG['hidden_size_2'],
                dropout_1=CONFIG['dropout_1'],
                dropout_2=CONFIG['dropout_2'],
            ).to(device)

            # Weighted loss
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=CONFIG['learning_rate'],
                weight_decay=CONFIG['weight_decay']
            )

            # Train
            best_state, history = train_one_model(
                model, train_loader, val_loader,
                data['val_emb'], data['val_labels'],
                criterion, optimizer,
                num_epochs=CONFIG['num_epochs'],
                patience=CONFIG['early_stopping_patience'],
                device=device
            )

            # Load best model
            model.load_state_dict(best_state)

            # Evaluate on test set
            model.eval()
            with torch.no_grad():
                test_probs = model.predict_proba(data['test_emb'].to(device)).cpu()

            metrics = compute_all_metrics(data['test_labels'], test_probs)
            print_metrics(metrics, "Test Results (Seed %d, t=%d°C)" % (seed, threshold))

            # Save per-seed results
            seed_dir = os.path.join(output_dir, "t%d" % threshold, "seed%d" % seed)
            os.makedirs(seed_dir, exist_ok=True)

            save_metrics(metrics, os.path.join(seed_dir, 'metrics.json'))
            torch.save(best_state, os.path.join(seed_dir, 'model.pt'))
            torch.save({
                'y_true': data['test_labels'],
                'y_prob': test_probs,
            }, os.path.join(seed_dir, 'predictions.pt'))

            plot_roc_curve(
                data['test_labels'], test_probs,
                title="ROC — Improved t=%d°C (seed=%d)" % (threshold, seed),
                save_path=os.path.join(seed_dir, 'roc_curve.png')
            )
            plot_training_history(history, save_path=os.path.join(seed_dir, 'training_history.png'))

            threshold_predictions.append(test_probs)
            threshold_results.append(metrics)

        # ── Ensemble results (average across seeds) ──
        ensemble_probs = torch.stack(threshold_predictions).mean(dim=0)
        ensemble_metrics = compute_all_metrics(data['test_labels'], ensemble_probs)
        print_metrics(ensemble_metrics, "ENSEMBLE Results (t=%d°C, %d seeds)" % (threshold, len(seeds)))

        ensemble_dir = os.path.join(output_dir, "t%d" % threshold, "ensemble")
        os.makedirs(ensemble_dir, exist_ok=True)
        save_metrics(ensemble_metrics, os.path.join(ensemble_dir, 'metrics.json'))
        torch.save({
            'y_true': data['test_labels'],
            'y_prob': ensemble_probs,
        }, os.path.join(ensemble_dir, 'predictions.pt'))

        plot_roc_curve(
            data['test_labels'], ensemble_probs,
            title="ROC — Improved Ensemble t=%d°C" % threshold,
            save_path=os.path.join(ensemble_dir, 'roc_curve.png')
        )
        plot_prc_curve(
            data['test_labels'], ensemble_probs,
            title="PRC — Improved Ensemble t=%d°C" % threshold,
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
    print("  V2 IMPROVED — FINAL SUMMARY (Ensemble)")
    print("=" * 70)
    print("  %-10s %-10s %-10s %-10s %-10s %-10s" % (
        'Threshold', 'AUC-ROC', 'AUC-PRC', 'F1', 'MCC', 'Bal.Acc'))
    print("  " + "-" * 60)
    for t in thresholds:
        m = all_results["t%d" % t]
        print("  %-10s %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f" % (
            "%d°C" % t, m['auc_roc'], m['auc_prc'], m['f1'], m['mcc'], m['balanced_accuracy']))
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='V2 Improved Training')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to prepared_data.pt')
    parser.add_argument('--thresholds', type=int, nargs='+', default=None,
                        help='Temperature thresholds (default: from config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    args = parser.parse_args()

    if args.epochs:
        CONFIG['num_epochs'] = args.epochs

    run_experiment(args)


if __name__ == '__main__':
    main()
