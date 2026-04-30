"""
Shared evaluation metrics for TemStaPro experiments.

Computes AUC-ROC, AUPRC, F1, MCC, Balanced Accuracy, and generates
comparison plots between experiment versions.
"""

import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    matthews_corrcoef, balanced_accuracy_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    classification_report
)


def compute_all_metrics(y_true, y_prob, threshold=0.5):
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: ground truth binary labels (numpy array or tensor)
        y_prob: predicted probabilities (numpy array or tensor)
        threshold: decision threshold for binary predictions

    Returns:
        dict with all metrics
    """
    # Convert to numpy
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_prob):
        y_prob = y_prob.cpu().numpy()

    y_true = y_true.astype(np.float64)
    y_prob = y_prob.astype(np.float64)

    y_pred = (y_prob >= threshold).astype(np.float64)

    metrics = {}

    # AUC-ROC
    try:
        metrics['auc_roc'] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics['auc_roc'] = 0.0

    # AUC-PRC (Average Precision)
    try:
        metrics['auc_prc'] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        metrics['auc_prc'] = 0.0

    # F1 Score
    metrics['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))

    # MCC
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics['mcc'] = float(matthews_corrcoef(y_true, y_pred))

    # Balanced Accuracy
    metrics['balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))

    # Standard Accuracy
    metrics['accuracy'] = float(np.mean(y_true == y_pred))

    # Confusion matrix derived metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    metrics['precision'] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    metrics['npv'] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

    # Counts
    metrics['tp'] = int(tp)
    metrics['fp'] = int(fp)
    metrics['tn'] = int(tn)
    metrics['fn'] = int(fn)
    metrics['n_positive'] = int(np.sum(y_true == 1))
    metrics['n_negative'] = int(np.sum(y_true == 0))
    metrics['n_total'] = int(len(y_true))

    return metrics


def print_metrics(metrics, title=""):
    """Pretty-print a metrics dictionary."""
    if title:
        print("\n" + "=" * 60)
        print("  %s" % title)
        print("=" * 60)

    print("  AUC-ROC:           %.4f" % metrics['auc_roc'])
    print("  AUC-PRC:           %.4f" % metrics['auc_prc'])
    print("  F1 (positive):     %.4f" % metrics['f1'])
    print("  F1 (macro):        %.4f" % metrics['f1_macro'])
    print("  MCC:               %.4f" % metrics['mcc'])
    print("  Balanced Accuracy: %.4f" % metrics['balanced_accuracy'])
    print("  Sensitivity:       %.4f  (TP=%d, FN=%d)" % (
        metrics['sensitivity'], metrics['tp'], metrics['fn']))
    print("  Specificity:       %.4f  (TN=%d, FP=%d)" % (
        metrics['specificity'], metrics['tn'], metrics['fp']))
    print("  Precision:         %.4f" % metrics['precision'])
    print("  Samples: %d total (%d pos, %d neg)" % (
        metrics['n_total'], metrics['n_positive'], metrics['n_negative']))


def save_metrics(metrics, filepath):
    """Save metrics dict to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Convert any numpy types to Python types
    clean_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (np.integer,)):
            clean_metrics[k] = int(v)
        elif isinstance(v, (np.floating,)):
            clean_metrics[k] = float(v)
        else:
            clean_metrics[k] = v

    with open(filepath, 'w') as f:
        json.dump(clean_metrics, f, indent=2)
    print("  Metrics saved to: %s" % filepath)


def load_metrics(filepath):
    """Load metrics dict from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


# ──────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────

def plot_roc_curve(y_true, y_prob, title="ROC Curve", save_path=None, label=None):
    """Plot a single ROC curve."""
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_prob):
        y_prob = y_prob.cpu().numpy()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 8))
    lbl = label or ("AUC = %.4f" % auc)
    ax.plot(fpr, tpr, linewidth=2, label=lbl, color='#6366f1')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print("  ROC curve saved: %s" % save_path)
    plt.close()
    return fig


def plot_prc_curve(y_true, y_prob, title="Precision-Recall Curve", save_path=None):
    """Plot a Precision-Recall curve."""
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_prob):
        y_prob = y_prob.cpu().numpy()

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(recall, precision, linewidth=2, label="AP = %.4f" % ap, color='#10b981')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print("  PRC curve saved: %s" % save_path)
    plt.close()
    return fig


def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss/metrics over epochs.

    Args:
        history: dict with keys like 'train_loss', 'val_loss', 'val_auc_roc', etc.
                 Each value is a list of per-epoch values.
    """
    n_plots = 0
    if 'train_loss' in history:
        n_plots += 1
    if 'val_auc_roc' in history:
        n_plots += 1
    if 'val_mcc' in history:
        n_plots += 1

    n_plots = max(n_plots, 1)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    idx = 0

    # Loss plot
    if 'train_loss' in history:
        ax = axes[idx]
        epochs = range(1, len(history['train_loss']) + 1)
        ax.plot(epochs, history['train_loss'], label='Train Loss', color='#6366f1', linewidth=2)
        if 'val_loss' in history:
            ax.plot(epochs, history['val_loss'], label='Val Loss', color='#f59e0b', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        idx += 1

    # AUC-ROC plot
    if 'val_auc_roc' in history:
        ax = axes[idx]
        epochs = range(1, len(history['val_auc_roc']) + 1)
        ax.plot(epochs, history['val_auc_roc'], label='Val AUC-ROC', color='#10b981', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AUC-ROC')
        ax.set_title('Validation AUC-ROC', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        idx += 1

    # MCC plot
    if 'val_mcc' in history:
        ax = axes[idx]
        epochs = range(1, len(history['val_mcc']) + 1)
        ax.plot(epochs, history['val_mcc'], label='Val MCC', color='#ef4444', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MCC')
        ax.set_title('Validation MCC', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        idx += 1

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print("  Training history saved: %s" % save_path)
    plt.close()
    return fig


def plot_comparison_roc(results_dict, title="ROC Curve Comparison", save_path=None):
    """
    Plot ROC curves from multiple experiments on the same axes.

    Args:
        results_dict: dict of {name: (y_true, y_prob)} pairs
        title: plot title
        save_path: where to save the plot
    """
    colors = ['#6366f1', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6']
    fig, ax = plt.subplots(figsize=(9, 9))

    for i, (name, (y_true, y_prob)) in enumerate(results_dict.items()):
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_prob):
            y_prob = y_prob.cpu().numpy()

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        color = colors[i % len(colors)]
        ax.plot(fpr, tpr, linewidth=2.5, label="%s (AUC=%.4f)" % (name, auc), color=color)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print("  Comparison ROC saved: %s" % save_path)
    plt.close()
    return fig


def plot_metrics_bar_comparison(metrics_list, labels, save_path=None):
    """
    Plot bar chart comparing key metrics across experiments.

    Args:
        metrics_list: list of metrics dicts
        labels: list of experiment names
        save_path: where to save the plot
    """
    metric_names = ['auc_roc', 'auc_prc', 'f1', 'mcc', 'balanced_accuracy', 'sensitivity', 'specificity']
    display_names = ['AUC-ROC', 'AUC-PRC', 'F1', 'MCC', 'Bal. Acc', 'Sensitivity', 'Specificity']
    colors = ['#6366f1', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6']

    x = np.arange(len(metric_names))
    width = 0.8 / len(labels)

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
        values = [metrics.get(m, 0) for m in metric_names]
        bars = ax.bar(x + i * width, values, width, label=label,
                      color=colors[i % len(colors)], alpha=0.85, edgecolor='white')

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                    '%.3f' % val, ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('Metric', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('Model Comparison — Key Metrics', fontsize=15, fontweight='bold')
    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels(display_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print("  Metrics comparison saved: %s" % save_path)
    plt.close()
    return fig
