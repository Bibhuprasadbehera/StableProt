"""
V4 Improved Regression — Training script.

Same OGT-only data as V3, but with all training improvements:
  - Huber loss
  - Target normalization
  - LR scheduler
  - Gradient clipping
  - Mixup augmentation
  - Per-sample bin weights
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, EXPERIMENTS_DIR)

from model import MLP_Regression_Improved
from config import CONFIG
from common.data_utils import TemStaProDataset
from torch.utils.data import DataLoader, WeightedRandomSampler


def get_binned_weights(temps, num_bins=8, min_val=0.0, max_val=100.0):
    """Inverse frequency weighting by temperature bin."""
    temps_np = np.array(temps)
    bins = np.linspace(min_val, max_val, num_bins + 1)
    inds = np.digitize(temps_np, bins) - 1
    inds = np.clip(inds, 0, num_bins - 1)
    counts = np.bincount(inds, minlength=num_bins)
    counts = np.where(counts == 0, 1, counts)
    weights_per_bin = len(temps_np) / (num_bins * counts)
    return torch.tensor(weights_per_bin[inds], dtype=torch.float32)


def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    idx = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[idx], lam * y + (1 - lam) * y[idx]


def create_data_loaders(data_path, batch_size, train_temps_for_weights=None):
    data = torch.load(data_path)

    train_labels = torch.tensor(data['train_temps'], dtype=torch.float32)
    val_labels = torch.tensor(data['val_temps'], dtype=torch.float32)
    test_labels = torch.tensor(data['test_temps'], dtype=torch.float32)

    train_dataset = TemStaProDataset(data['train_embeddings'], train_labels)
    val_dataset = TemStaProDataset(data['val_embeddings'], val_labels)
    test_dataset = TemStaProDataset(data['test_embeddings'], test_labels)

    # Weighted sampler for imbalanced temperature distribution
    sample_weights = get_binned_weights(data['train_temps'])
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, data['test_temps'], data['train_temps']


def evaluate(model, loader, criterion, device, mean=0.0, std=1.0, denorm=False):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device).float(), targets.to(device).float()
            outputs = model(inputs).squeeze(-1)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            preds = outputs.cpu()
            tgts = targets.cpu()
            if denorm:
                preds = preds * std + mean
                tgts = tgts * std + mean

            all_preds.extend(preds.tolist())
            all_targets.extend(tgts.tolist())

    avg_loss = total_loss / len(loader.dataset)
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
    return avg_loss, mae


def train_one_seed(seed, train_loader, val_loader, train_mean, train_std, save_dir, device):
    print("\n" + "=" * 50)
    print(f"Training V4 Improved Regression (Seed {seed})")
    print("=" * 50)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = MLP_Regression_Improved(
        input_size=CONFIG['input_size'],
        hidden_size_1=CONFIG['hidden_size_1'],
        hidden_size_2=CONFIG['hidden_size_2'],
        dropout_1=CONFIG['dropout_1'],
        dropout_2=CONFIG['dropout_2']
    ).to(device)

    # Huber loss instead of MSE
    criterion = nn.HuberLoss(delta=CONFIG['huber_delta'])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=CONFIG['lr_scheduler_patience'],
        factor=CONFIG['lr_scheduler_factor'], min_lr=1e-6
    )

    best_val_mae = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(save_dir, 'model.pt')
    history = {'train_loss': [], 'val_loss': [], 'val_mae': []}

    for epoch in range(CONFIG['num_epochs']):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device).float(), targets.to(device).float()

            # Target normalization
            if CONFIG['target_normalization']:
                targets = (targets - train_mean) / train_std

            # Mixup
            if CONFIG['mixup_alpha'] > 0:
                inputs, targets = mixup_data(inputs, targets, CONFIG['mixup_alpha'])

            optimizer.zero_grad()
            outputs = model(inputs).squeeze(-1)
            loss = criterion(outputs, targets)
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip_max_norm'])
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation (denormalize for MAE)
        val_loss, val_mae = evaluate(model, val_loader, criterion, device,
                                      mean=train_mean, std=train_std,
                                      denorm=CONFIG['target_normalization'])
        scheduler.step(val_mae)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)

        print(f"Epoch {epoch+1:2d}/{CONFIG['num_epochs']} | Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.4f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("  -> Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Plot training history
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(history['val_mae'], label='Val MAE', color='green')
    plt.legend()
    plt.title('Mean Absolute Error')
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()

    return best_model_path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None)
    args = parser.parse_args()

    data_path = args.data if args.data else os.path.join(EXPERIMENTS_DIR, CONFIG['data_path'])
    if not os.path.exists(data_path):
        data_path = os.path.join(EXPERIMENTS_DIR, 'prepared_data.pt')
        if not os.path.exists(data_path):
            print(f"ERROR: No data at {data_path}")
            sys.exit(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    results_dir = os.path.join(SCRIPT_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)

    train_loader, val_loader, test_loader, test_temps, train_temps = create_data_loaders(
        data_path, CONFIG['batch_size']
    )

    train_mean = np.mean(train_temps)
    train_std = np.std(train_temps)
    print(f"Train normalization: mean={train_mean:.2f}, std={train_std:.2f}")

    # Normalize val/test loader targets if needed
    # (handled inside evaluate() with denorm flag)

    ensemble_preds = []

    for seed in CONFIG['seeds']:
        seed_dir = os.path.join(results_dir, f"seed{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        best_model_path = train_one_seed(seed, train_loader, val_loader,
                                          train_mean, train_std, seed_dir, device)

        # Test evaluation
        model = MLP_Regression_Improved(
            input_size=CONFIG['input_size'],
            hidden_size_1=CONFIG['hidden_size_1'],
            hidden_size_2=CONFIG['hidden_size_2']
        ).to(device)
        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        preds = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device).float()
                outputs = model(inputs).squeeze(-1)
                if CONFIG['target_normalization']:
                    outputs = outputs * train_std + train_mean
                preds.extend(outputs.cpu().tolist())

        ensemble_preds.append(preds)

        torch.save({
            'y_true': torch.tensor(test_temps),
            'y_pred': torch.tensor(preds)
        }, os.path.join(seed_dir, 'predictions.pt'))

        mae = np.mean(np.abs(np.array(preds) - np.array(test_temps)))
        with open(os.path.join(seed_dir, 'metrics.json'), 'w') as f:
            json.dump({'mae': float(mae)}, f, indent=2)
        print(f"Seed {seed} Test MAE: {mae:.4f}")

    # Ensemble
    ensemble_dir = os.path.join(results_dir, "ensemble")
    os.makedirs(ensemble_dir, exist_ok=True)

    mean_preds = np.mean(ensemble_preds, axis=0)
    mae = np.mean(np.abs(mean_preds - np.array(test_temps)))

    torch.save({
        'y_true': torch.tensor(test_temps),
        'y_pred': torch.tensor(mean_preds)
    }, os.path.join(ensemble_dir, 'predictions.pt'))

    with open(os.path.join(ensemble_dir, 'metrics.json'), 'w') as f:
        json.dump({'mae': float(mae)}, f, indent=2)

    print(f"\n{'='*50}")
    print(f"V4 IMPROVED ENSEMBLE MAE: {mae:.4f} °C")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
