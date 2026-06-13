#!/usr/bin/env python3
"""
Phase 4: Optuna Hyperparameter Search for V7

Searches over key hyperparameters using 1 seed per trial.
Top 3 configs retrained with 5 seeds in Phase 3.

Usage:
    python phase4_optuna_search.py [--n-trials 50] [--device cuda]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_FILE = PROJECT_ROOT / "data" / "embeddings" / "prepared_data_v7_saprot1.3b_seqonly.pt"
STUDY_DIR = PROJECT_ROOT / "experiments" / "src" / "training" / "v7_shared" / "optuna_study"

# Import model from Phase 3
import sys
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "src" / "training" / "v7_shared"))
from train import MultiHeadSaProtV7, SimpleDataset, compute_sample_weights


def objective(trial, data, device):
    """Optuna objective: train V7 with sampled hyperparameters, return val MAE."""

    # Sample hyperparameters
    config = {
        'input_dim': 1280,
        'hidden1': trial.suggest_categorical('hidden1', [256, 384, 512, 768]),
        'hidden2': trial.suggest_categorical('hidden2', [128, 192, 256, 384]),
        'dropout1': trial.suggest_float('dropout1', 0.1, 0.5, step=0.05),
        'dropout2': trial.suggest_float('dropout2', 0.1, 0.5, step=0.05),
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'ogt_loss_scale': trial.suggest_float('ogt_loss_scale', 0.01, 0.1, log=True),
        'weight_clamp_max': trial.suggest_float('weight_clamp_max', 4.0, 12.0, step=1.0),
        'epochs': 100,
        'early_stopping_patience': 12,
        'bin_width': 5,
        'bin_range': (25, 100),
        'scheduler_T0': 10,
        'scheduler_Tmult': 2,
    }

    # Ensure hidden2 <= hidden1
    if config['hidden2'] > config['hidden1']:
        raise optuna.TrialPruned()

    seed = 42  # Fixed seed for fair comparison
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build model
    model = MultiHeadSaProtV7(
        input_dim=config['input_dim'],
        hidden1=config['hidden1'],
        hidden2=config['hidden2'],
        dropout1=config['dropout1'],
        dropout2=config['dropout2'],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'],
                           weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config['scheduler_T0'], T_mult=config['scheduler_Tmult'],
        eta_min=1e-6
    )
    huber = nn.HuberLoss(reduction='none', delta=1.0)

    # DataLoaders
    tm_loader = DataLoader(
        SimpleDataset(data['train_tm']['embeddings'], data['train_tm_labels']),
        batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=True
    )
    ogt_loader = DataLoader(
        SimpleDataset(data['train_ogt']['embeddings'], data['ogt_labels']),
        batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        SimpleDataset(data['val_tm']['embeddings'], data['val_tm_labels']),
        batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=True
    )

    # Compute bin weights
    bins = np.arange(config['bin_range'][0], config['bin_range'][1] + config['bin_width'], config['bin_width'])
    labels_np = data['train_tm_labels'].numpy()
    bin_indices = np.digitize(labels_np, bins) - 1
    bin_counts = np.zeros(len(bins) - 1)
    for b in range(len(bins) - 1):
        bin_counts[b] = max(1, np.sum(bin_indices == b))
    median_count = np.median(bin_counts[bin_counts > 0])
    bin_weights = np.minimum(np.sqrt(median_count / bin_counts), config['weight_clamp_max'])
    tm_weights_map = {b: float(bin_weights[b]) for b in range(len(bin_weights))}

    # Training loop
    best_val_mae = float('inf')
    patience_counter = 0

    for epoch in range(config['epochs']):
        model.train()
        ogt_iter = iter(ogt_loader)

        for tm_x, tm_y in tm_loader:
            tm_x, tm_y = tm_x.to(device), tm_y.to(device)

            optimizer.zero_grad()

            # Tm step
            tm_pred = model(tm_x, task='tm')
            tm_loss_raw = huber(tm_pred, tm_y)
            tm_y_np = tm_y.cpu().numpy()
            bin_idx = np.digitize(tm_y_np, bins) - 1
            batch_weights = torch.tensor(
                [tm_weights_map.get(int(bi), 1.0) for bi in bin_idx],
                device=device, dtype=torch.float32
            )
            tm_loss = (tm_loss_raw * batch_weights).mean()
            tm_loss.backward()

            # OGT step
            try:
                ogt_x, ogt_y = next(ogt_iter)
            except StopIteration:
                ogt_iter = iter(ogt_loader)
                ogt_x, ogt_y = next(ogt_iter)

            ogt_x, ogt_y = ogt_x.to(device), ogt_y.to(device)
            ogt_pred = model(ogt_x, task='ogt')
            ogt_loss = huber(ogt_pred, ogt_y).mean() * config['ogt_loss_scale']
            ogt_loss.backward()

            optimizer.step()

        scheduler.step()

        # Validation
        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x, task='tm')
                val_mae += torch.abs(pred - y).sum().item()
        val_mae /= len(val_loader.dataset)

        # Report to Optuna for pruning
        trial.report(val_mae, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                break

    return best_val_mae


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Optuna Hyperparameter Search")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("Phase 4: Optuna Hyperparameter Search")
    STUDY_DIR.mkdir(parents=True, exist_ok=True)

    # Load data once
    print("Loading data...")
    raw_data = torch.load(DATA_FILE, map_location='cpu', weights_only=False)

    # Pre-process labels
    data = {
        'train_tm': raw_data['train_tm'],
        'val_tm': raw_data['val_tm'],
        'train_ogt': raw_data['train_ogt'],
    }
    data['train_tm_labels'] = raw_data['train_tm']['tm_consensus'].float() \
        if isinstance(raw_data['train_tm']['tm_consensus'], torch.Tensor) \
        else torch.tensor(raw_data['train_tm']['tm_consensus']).float()
    data['val_tm_labels'] = raw_data['val_tm']['tm_consensus'].float() \
        if isinstance(raw_data['val_tm']['tm_consensus'], torch.Tensor) \
        else torch.tensor(raw_data['val_tm']['tm_consensus']).float()
    data['ogt_labels'] = raw_data['train_ogt']['ogt_consensus'].float()

    print(f"  Train Tm: {len(data['train_tm_labels'])}")
    print(f"  Val Tm:   {len(data['val_tm_labels'])}")
    print(f"  OGT:      {len(data['ogt_labels'])}")

    # Create Optuna study
    study = optuna.create_study(
        direction="minimize",
        study_name="stableprot_v7",
        storage=f"sqlite:///{STUDY_DIR / 'optuna.db'}",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20),
    )

    study.optimize(
        lambda trial: objective(trial, data, args.device),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    # Results
    print(f"\n{'='*60}")
    print(f"  Optuna Search Complete")
    print(f"{'='*60}")
    print(f"  Best val MAE: {study.best_value:.4f}")
    print(f"  Best params: {json.dumps(study.best_params, indent=2)}")

    # Save top 3 configs
    top_trials = sorted(study.trials, key=lambda t: t.value if t.value else float('inf'))[:3]
    top_configs = []
    for i, trial in enumerate(top_trials):
        config = trial.params.copy()
        config['val_mae'] = trial.value
        config['trial_number'] = trial.number
        top_configs.append(config)
        print(f"\n  Top {i+1}: MAE={trial.value:.4f} | {trial.params}")

    with open(STUDY_DIR / "top3_configs.json", 'w') as f:
        json.dump(top_configs, f, indent=2)
    print(f"\n  Top 3 configs saved to {STUDY_DIR / 'top3_configs.json'}")
    print(f"  Next: retrain each with 5 seeds using phase3_v7_train.py --config <config.json>")


if __name__ == "__main__":
    main()
