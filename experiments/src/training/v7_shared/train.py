#!/usr/bin/env python3
"""
Phase 3: V7 Shared Backbone Training

Multi-head architecture with shared backbone for OGT + Tm prediction.
Both heads use SaProt 1.3B sequence-only embeddings (2560-dim).

Key design:
  - Shared backbone: 2 layers (2560 → H1 → H2)
  - Independent heads: OGT (H2 → 1), Tm (H2 → 1)
  - OGT loss scaled by 0.03 (prevents gradient domination from 33:1 data ratio)
  - Tm weighted Huber: min(sqrt(median/count), 8.0) with 5°C bins
  - 5-seed ensemble with CosineAnnealingWarmRestarts

Usage:
    python phase3_v7_train.py [--config CONFIG_JSON] [--seeds 1,2,3,4,5]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_FILE = PROJECT_ROOT / "data" / "embeddings" / "prepared_data_v7_saprot1.3b_seqonly.pt"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "src" / "training" / "v7_shared" / "results"

# ── Default Config ──
CONFIG = {
    'input_dim': 1280,       # SaProt 1.3B output dim
    'hidden1': 512,
    'hidden2': 256,
    'dropout1': 0.3,
    'dropout2': 0.2,
    'lr': 1e-4,
    'weight_decay': 1e-5,
    'batch_size': 64,
    'epochs': 150,
    'early_stopping_patience': 15,
    'ogt_loss_scale': 0.03,   # Scale OGT loss to prevent gradient domination
    'bin_width': 5,           # °C for temperature bins
    'bin_range': (25, 100),
    'weight_clamp_max': 8.0,  # Max weight for rare bins
    'scheduler_T0': 10,
    'scheduler_Tmult': 2,
}


# ── Model ──
class MultiHeadSaProtV7(nn.Module):
    """Shared backbone with independent OGT and Tm heads.
    
    Architecture:
        SaProt 1.3B (frozen, 2560-dim) → Shared MLP → {OGT head, Tm head}
    """
    def __init__(self, input_dim=2560, hidden1=512, hidden2=256,
                 dropout1=0.3, dropout2=0.2):
        super().__init__()

        # Shared backbone
        self.shared_layer1 = nn.Linear(input_dim, hidden1)
        self.shared_bn1 = nn.BatchNorm1d(hidden1)
        self.shared_layer2 = nn.Linear(hidden1, hidden2)
        self.shared_bn2 = nn.BatchNorm1d(hidden2)
        self.shared_residual = nn.Linear(hidden1, hidden2)

        # Task-specific heads
        self.head_tm = nn.Linear(hidden2, 1)
        self.head_ogt = nn.Linear(hidden2, 1)

        # Shared dropout
        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)

    def forward(self, x, task='tm'):
        # Shared backbone
        x1 = self.dropout1(torch.relu(self.shared_bn1(self.shared_layer1(x))))
        x2 = self.dropout2(torch.relu(
            self.shared_bn2(self.shared_layer2(x1)) + self.shared_residual(x1)
        ))

        # Task-specific head
        if task == 'tm':
            return self.head_tm(x2).squeeze(-1)
        elif task == 'ogt':
            return self.head_ogt(x2).squeeze(-1)
        else:
            raise ValueError(f"Unknown task: {task}")


# ── Datasets ──
class SimpleDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings.float()
        self.labels = labels.float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


# ── Weighting ──
def compute_sample_weights(labels, bin_width=5, bin_range=(25, 100), clamp_max=8.0):
    """Compute per-sample weights using sqrt(median/count) with 5°C bins.
    
    Returns: weight tensor of same length as labels
    """
    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
    bins = np.arange(bin_range[0], bin_range[1] + bin_width, bin_width)
    bin_indices = np.digitize(labels_np, bins) - 1

    # Count per bin
    bin_counts = np.zeros(len(bins) - 1)
    for b in range(len(bins) - 1):
        bin_counts[b] = max(1, np.sum(bin_indices == b))

    median_count = np.median(bin_counts[bin_counts > 0])

    # sqrt(median/count), clamped
    bin_weights = np.minimum(np.sqrt(median_count / bin_counts), clamp_max)

    # Assign per-sample weights
    sample_weights = np.ones(len(labels_np))
    for i, bi in enumerate(bin_indices):
        if 0 <= bi < len(bin_weights):
            sample_weights[i] = bin_weights[bi]

    # Print weight table
    print(f"\n  Temperature bin weights (5°C bins, clamp={clamp_max}):")
    print(f"  {'Bin':>10} | {'Count':>6} | {'Weight':>6}")
    print(f"  {'-'*10}-+-{'-'*6}-+-{'-'*6}")
    for b in range(len(bins) - 1):
        if bin_counts[b] > 0:
            print(f"  {bins[b]:>4}-{bins[b+1]:<4}  | {int(bin_counts[b]):>6} | {bin_weights[b]:>6.2f}")

    return torch.tensor(sample_weights, dtype=torch.float32)


# ── Training ──
def train_one_seed(seed, config, tm_loader, ogt_loader, val_loader,
                   tm_weights_map, device, save_dir):
    """Train single seed of V7 model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MultiHeadSaProtV7(
        input_dim=config['input_dim'],
        hidden1=config['hidden1'],
        hidden2=config['hidden2'],
        dropout1=config['dropout1'],
        dropout2=config['dropout2'],
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config['scheduler_T0'], T_mult=config['scheduler_Tmult'],
        eta_min=1e-6
    )

    huber = nn.HuberLoss(reduction='none', delta=1.0)
    ogt_scale = config['ogt_loss_scale']

    best_val_mae = float('inf')
    patience_counter = 0
    best_model_path = save_dir / "best_model.pt"

    for epoch in range(config['epochs']):
        model.train()
        epoch_tm_loss = 0.0
        epoch_ogt_loss = 0.0
        tm_batches = 0
        ogt_batches = 0

        # Alternating optimization: interleave OGT and Tm batches
        ogt_iter = iter(ogt_loader)
        for tm_x, tm_y in tm_loader:
            tm_x, tm_y = tm_x.to(device), tm_y.to(device)

            # --- Tm step ---
            optimizer.zero_grad()
            tm_pred = model(tm_x, task='tm')
            tm_loss_raw = huber(tm_pred, tm_y)

            # Apply per-sample weights
            batch_weights = torch.tensor(
                [tm_weights_map.get(int(idx), 1.0) for idx in range(len(tm_y))],
                device=device, dtype=torch.float32
            )
            # Actually use the label-based weights
            bins = np.arange(config['bin_range'][0], config['bin_range'][1] + config['bin_width'], config['bin_width'])
            tm_y_np = tm_y.cpu().numpy()
            bin_idx = np.digitize(tm_y_np, bins) - 1
            batch_weights = torch.tensor(
                [tm_weights_map.get(int(bi), 1.0) for bi in bin_idx],
                device=device, dtype=torch.float32
            )

            tm_loss = (tm_loss_raw * batch_weights).mean()
            tm_loss.backward()

            # --- OGT step ---
            try:
                ogt_x, ogt_y = next(ogt_iter)
            except StopIteration:
                ogt_iter = iter(ogt_loader)
                ogt_x, ogt_y = next(ogt_iter)

            ogt_x, ogt_y = ogt_x.to(device), ogt_y.to(device)
            ogt_pred = model(ogt_x, task='ogt')
            ogt_loss = huber(ogt_pred, ogt_y).mean() * ogt_scale
            ogt_loss.backward()

            optimizer.step()

            epoch_tm_loss += tm_loss.item()
            epoch_ogt_loss += ogt_loss.item()
            tm_batches += 1
            ogt_batches += 1

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        # Validation (Tm only)
        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x, task='tm')
                val_mae += torch.abs(pred - y).sum().item()
        val_mae /= len(val_loader.dataset)

        avg_tm_loss = epoch_tm_loss / max(tm_batches, 1)
        avg_ogt_loss = epoch_ogt_loss / max(ogt_batches, 1)

        print(f"  Epoch {epoch+1:>3} | Tm Loss: {avg_tm_loss:.4f} | "
              f"OGT Loss: {avg_ogt_loss:.4f} | Val MAE: {val_mae:.4f} | LR: {lr:.6f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  → Saved best model (MAE: {val_mae:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    return best_model_path, best_val_mae


def main():
    parser = argparse.ArgumentParser(description="Phase 3: V7 Shared Backbone Training")
    parser.add_argument("--config", type=str, help="Path to config JSON (overrides defaults)")
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5", help="Comma-separated seeds")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = CONFIG.copy()
    if args.config:
        with open(args.config) as f:
            config.update(json.load(f))
        print(f"Loaded config from {args.config}")

    seeds = [int(s) for s in args.seeds.split(",")]
    device = args.device

    print("Phase 3: V7 Shared Backbone Training")
    print(f"  Config: {json.dumps(config, indent=2)}")
    print(f"  Seeds: {seeds}")
    print(f"  Device: {device}")

    # Load data
    print("\nLoading data...")
    data = torch.load(DATA_FILE, map_location='cpu', weights_only=False)

    train_tm_emb = data['train_tm']['embeddings']
    train_tm_lbl = torch.tensor(data['train_tm']['tm_consensus']).float() \
        if not isinstance(data['train_tm']['tm_consensus'], torch.Tensor) \
        else data['train_tm']['tm_consensus'].float()

    val_tm_emb = data['val_tm']['embeddings']
    val_tm_lbl = torch.tensor(data['val_tm']['tm_consensus']).float() \
        if not isinstance(data['val_tm']['tm_consensus'], torch.Tensor) \
        else data['val_tm']['tm_consensus'].float()

    ogt_emb = data['train_ogt']['embeddings']
    ogt_lbl = data['train_ogt']['ogt_consensus'].float()

    print(f"  Train Tm: {train_tm_emb.shape}")
    print(f"  Val Tm:   {val_tm_emb.shape}")
    print(f"  OGT:      {ogt_emb.shape}")

    # Compute Tm sample weights
    print("\nComputing temperature bin weights...")
    sample_weights = compute_sample_weights(
        train_tm_lbl,
        bin_width=config['bin_width'],
        bin_range=config['bin_range'],
        clamp_max=config['weight_clamp_max']
    )

    # Build bin → weight mapping for efficient lookup during training
    bins = np.arange(config['bin_range'][0], config['bin_range'][1] + config['bin_width'], config['bin_width'])
    labels_np = train_tm_lbl.numpy()
    bin_indices = np.digitize(labels_np, bins) - 1
    bin_counts = np.zeros(len(bins) - 1)
    for b in range(len(bins) - 1):
        bin_counts[b] = max(1, np.sum(bin_indices == b))
    median_count = np.median(bin_counts[bin_counts > 0])
    bin_weights = np.minimum(np.sqrt(median_count / bin_counts), config['weight_clamp_max'])
    tm_weights_map = {b: float(bin_weights[b]) for b in range(len(bin_weights))}

    # DataLoaders
    tm_loader = DataLoader(
        SimpleDataset(train_tm_emb, train_tm_lbl),
        batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True
    )
    ogt_loader = DataLoader(
        SimpleDataset(ogt_emb, ogt_lbl),
        batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        SimpleDataset(val_tm_emb, val_tm_lbl),
        batch_size=config['batch_size'], shuffle=False, num_workers=2, pin_memory=True
    )

    # Train ensemble
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  Training seed {seed}")
        print(f"{'='*60}")

        seed_dir = RESULTS_DIR / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        best_path, best_mae = train_one_seed(
            seed, config, tm_loader, ogt_loader, val_loader,
            tm_weights_map, device, seed_dir
        )

        # Evaluate on test set
        model = MultiHeadSaProtV7(
            input_dim=config['input_dim'],
            hidden1=config['hidden1'],
            hidden2=config['hidden2'],
        ).to(device)
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        model.eval()

        test_tm_emb = data['test_tm']['embeddings'].to(device)
        test_tm_lbl = data['test_tm']['tm_consensus'].float().to(device) \
            if isinstance(data['test_tm']['tm_consensus'], torch.Tensor) \
            else torch.tensor(data['test_tm']['tm_consensus']).float().to(device)

        with torch.no_grad():
            test_pred = model(test_tm_emb.float(), task='tm')
            test_mae = torch.abs(test_pred - test_tm_lbl).mean().item()

        result = {'seed': seed, 'val_mae': best_mae, 'test_mae': test_mae}
        all_results.append(result)
        print(f"\n  Seed {seed}: Val MAE = {best_mae:.4f}, Test MAE = {test_mae:.4f}")

        # Save per-seed metrics
        with open(seed_dir / "metrics.json", 'w') as f:
            json.dump(result, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Ensemble Summary")
    print(f"{'='*60}")
    val_maes = [r['val_mae'] for r in all_results]
    test_maes = [r['test_mae'] for r in all_results]
    print(f"  Val MAE:  {np.mean(val_maes):.4f} ± {np.std(val_maes):.4f}")
    print(f"  Test MAE: {np.mean(test_maes):.4f} ± {np.std(test_maes):.4f}")

    # Save config
    with open(RESULTS_DIR / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Save ensemble summary
    with open(RESULTS_DIR / "ensemble_summary.json", 'w') as f:
        json.dump({
            'config': config,
            'results': all_results,
            'ensemble_val_mae': float(np.mean(val_maes)),
            'ensemble_test_mae': float(np.mean(test_maes)),
        }, f, indent=2)


if __name__ == "__main__":
    main()
