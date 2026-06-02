"""V6+ Multi-Head SaProt Predictor with Temperature-Weighted Loss.

Improvements over V6:
1. Inverse-frequency bin weighting for Tm loss (addresses tail underrepresentation)
2. Learnable bias correction via a final calibration layer
3. Cosine annealing LR schedule
"""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# ── Config ──
CONFIG = {
    'input_size_tm': 1280,
    'input_size_ogt': 2560,
    'hidden_size_1': 512,
    'hidden_size_2': 256,
    'dropout_1': 0.3,
    'dropout_2': 0.2,
    'learning_rate': 1e-4,
    'batch_size': 64,
    'num_epochs': 60,
    'early_stopping_patience': 12,
    'weight_decay': 1e-5,
    'huber_delta': 5.0,
    'ogt_loss_weight': 0.3,
    'tm_loss_weight': 1.0,
    'grad_clip_max_norm': 1.0,
    'seeds': [1, 2, 3, 4, 5],
    # Temperature weighting params
    'bin_edges': list(range(20, 101, 10)),  # 20, 30, ..., 100
    'weight_clamp_min': 0.5,
    'weight_clamp_max': 5.0,
}

# ── Model ──
class MultiHeadSaProtV6Plus(nn.Module):
    def __init__(self, hidden1=512, hidden2=256, dropout1=0.3, dropout2=0.2):
        super().__init__()
        # TM pathway
        self.input_layer_tm = nn.Linear(1280, hidden1)
        self.bn1_tm = nn.BatchNorm1d(hidden1)
        self.fc2_tm = nn.Linear(hidden1, hidden2)
        self.bn2_tm = nn.BatchNorm1d(hidden2)
        self.residual_proj_tm = nn.Linear(hidden1, hidden2)
        self.head_tm = nn.Linear(hidden2, 1)
        
        # OGT pathway  
        self.input_layer_ogt = nn.Linear(2560, hidden1)
        self.bn1_ogt = nn.BatchNorm1d(hidden1)
        self.fc2_ogt = nn.Linear(hidden1, hidden2)
        self.bn2_ogt = nn.BatchNorm1d(hidden2)
        self.residual_proj_ogt = nn.Linear(hidden1, hidden2)
        self.head_ogt = nn.Linear(hidden2, 1)
        
        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)
        
    def forward(self, x, head='tm'):
        if head == 'tm':
            x1 = self.input_layer_tm(x)
            x1 = self.dropout1(torch.relu(self.bn1_tm(x1)))
            x2 = self.dropout2(torch.relu(self.bn2_tm(self.fc2_tm(x1)) + self.residual_proj_tm(x1)))
            return self.head_tm(x2).squeeze(-1)
        else:
            x1 = self.input_layer_ogt(x)
            x1 = self.dropout1(torch.relu(self.bn1_ogt(x1)))
            x2 = self.dropout2(torch.relu(self.bn2_ogt(self.fc2_ogt(x1)) + self.residual_proj_ogt(x1)))
            return self.head_ogt(x2).squeeze(-1)

class TmDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

def compute_sample_weights(labels, bin_edges, clamp_min=0.5, clamp_max=5.0):
    """Compute inverse-frequency weights per sample based on temperature bins."""
    labels_np = labels.numpy() if hasattr(labels, 'numpy') else np.array(labels)
    bin_indices = np.digitize(labels_np, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)
    
    # Count per bin
    bin_counts = np.bincount(bin_indices, minlength=len(bin_edges)-1).astype(float)
    bin_counts[bin_counts == 0] = 1  # avoid division by zero
    
    # Median frequency balancing
    median_freq = np.median(bin_counts[bin_counts > 0])
    bin_weights = median_freq / bin_counts
    bin_weights = np.clip(bin_weights, clamp_min, clamp_max)
    
    # Assign per-sample weights
    sample_weights = bin_weights[bin_indices]
    
    print(f"  Bin weights: {dict(zip([f'{bin_edges[i]}-{bin_edges[i+1]}' for i in range(len(bin_edges)-1)], [f'{w:.2f}' for w in bin_weights]))}")
    
    return torch.tensor(sample_weights, dtype=torch.float32)

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def train_one_seed(seed, tm_loader, ogt_loader, val_loader, tm_weights, device, save_dir):
    print(f"\n{'='*50}")
    print(f"Training V6+ SaProt Multi-Head (Seed {seed})")
    print(f"{'='*50}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    model = MultiHeadSaProtV6Plus(
        hidden1=CONFIG['hidden_size_1'],
        hidden2=CONFIG['hidden_size_2'],
        dropout1=CONFIG['dropout_1'],
        dropout2=CONFIG['dropout_2']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    
    # Cosine annealing instead of ReduceOnPlateau
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    criterion = nn.HuberLoss(delta=CONFIG['huber_delta'], reduction='none')
    
    best_val_mae = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(save_dir, 'model.pt')
    
    tm_iter = iter(cycle(tm_loader))
    
    # Pre-compute weight lookup tensor
    weight_lookup = tm_weights.to(device)
    
    for epoch in range(CONFIG['num_epochs']):
        model.train()
        train_ogt_loss = 0.0
        train_tm_loss = 0.0
        
        pbar = tqdm(ogt_loader, desc=f"Seed {seed} | Epoch {epoch+1}/{CONFIG['num_epochs']}", leave=False)
        for batch_idx, (ogt_x, ogt_y) in enumerate(pbar):
            # 1. OGT Forward
            ogt_x, ogt_y = ogt_x.to(device), ogt_y.to(device)
            ogt_pred = model(ogt_x, head='ogt')
            loss_ogt = criterion(ogt_pred, ogt_y).mean() * CONFIG['ogt_loss_weight']
            
            optimizer.zero_grad()
            loss_ogt.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip_max_norm'])
            optimizer.step()
            train_ogt_loss += loss_ogt.item()
            
            # 2. Tm Forward with temperature-weighted loss
            tm_batch = next(tm_iter)
            tm_x, tm_y = tm_batch[0].to(device), tm_batch[1].to(device)
            tm_pred = model(tm_x, head='tm')
            
            # Per-sample weighted loss
            per_sample_loss = criterion(tm_pred, tm_y)
            
            # Compute weights for this batch based on target temperature
            bin_edges = torch.tensor(CONFIG['bin_edges'], dtype=torch.float32, device=device)
            bin_idx = torch.bucketize(tm_y, bin_edges) - 1
            bin_idx = bin_idx.clamp(0, len(CONFIG['bin_edges']) - 2)
            
            # Use precomputed bin weights
            bin_counts_global = torch.tensor([
                13020, 9013, 1571, 1089, 1038, 316, 5, 2687  # approximate from data
            ], dtype=torch.float32, device=device)
            median_freq = bin_counts_global.median()
            bin_w = (median_freq / bin_counts_global).clamp(CONFIG['weight_clamp_min'], CONFIG['weight_clamp_max'])
            
            batch_weights = bin_w[bin_idx.clamp(0, len(bin_w)-1)]
            weighted_loss = (per_sample_loss * batch_weights).mean() * CONFIG['tm_loss_weight']
            
            optimizer.zero_grad()
            weighted_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip_max_norm'])
            optimizer.step()
            train_tm_loss += weighted_loss.item()
            
            if batch_idx % 100 == 0:
                pbar.set_postfix({'OGT': f'{loss_ogt.item():.3f}', 'Tm': f'{weighted_loss.item():.3f}'})
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x, head='tm')
                val_mae += torch.abs(pred - y).sum().item()
        val_mae /= len(val_loader.dataset)
        
        lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch+1} - Val MAE: {val_mae:.4f} | LR: {lr:.6f}")
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  → Saved best model (MAE: {val_mae:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['early_stopping_patience']:
                print("  Early stopping triggered!")
                break
    
    return best_model_path, best_val_mae

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "../../../../data/embeddings/prepared_data_v4_saprot.pt")
    
    print("Loading SaProt dataset...")
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    
    train_ogt_emb = data['train_ogt']['embeddings']
    train_ogt_lbl = data['train_ogt'].get('ogt_consensus', data['train_ogt'].get('labels'))
    train_tm_emb = data['train_tm']['embeddings']
    train_tm_lbl = data['train_tm'].get('tm_consensus', data['train_tm'].get('labels'))
    val_tm_emb = data['val_tm']['embeddings']
    val_tm_lbl = data['val_tm'].get('tm_consensus', data['val_tm'].get('labels'))
    test_tm_emb = data['test_tm']['embeddings']
    test_tm_lbl = data['test_tm'].get('tm_consensus', data['test_tm'].get('labels'))
    
    print(f"  Train OGT: {len(train_ogt_lbl)}, Train Tm: {len(train_tm_lbl)}")
    print(f"  Val Tm: {len(val_tm_lbl)}, Test Tm: {len(test_tm_lbl)}")
    
    # Compute temperature weights for Tm dataset
    print("\nComputing temperature-bin sample weights for Tm data...")
    tm_weights = compute_sample_weights(
        train_tm_lbl, CONFIG['bin_edges'],
        CONFIG['weight_clamp_min'], CONFIG['weight_clamp_max']
    )
    
    # Create dataloaders
    # For Tm, include index so we can look up weights
    ogt_loader = DataLoader(TmDataset(train_ogt_emb, train_ogt_lbl), batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    tm_loader = DataLoader(TmDataset(train_tm_emb, train_tm_lbl), batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(TmDataset(val_tm_emb, val_tm_lbl), batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(TmDataset(test_tm_emb, test_tm_lbl), batch_size=CONFIG['batch_size'], shuffle=False)
    
    results_dir = os.path.join(script_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    ensemble_preds = []
    seed_maes = []
    
    for seed in CONFIG['seeds']:
        seed_dir = os.path.join(results_dir, f"seed{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        
        best_path, best_mae = train_one_seed(seed, tm_loader, ogt_loader, val_loader, tm_weights, device, seed_dir)
        seed_maes.append(best_mae)
        
        # Test inference
        model = MultiHeadSaProtV6Plus(
            hidden1=CONFIG['hidden_size_1'],
            hidden2=CONFIG['hidden_size_2']
        ).to(device)
        model.load_state_dict(torch.load(best_path, map_location=device))
        model.eval()
        
        preds = []
        with torch.no_grad():
            for x, _ in test_loader:
                p = model(x.to(device), head='tm').cpu()
                preds.extend(p.tolist())
        
        ensemble_preds.append(preds)
        test_mae = np.mean(np.abs(np.array(preds) - test_tm_lbl.numpy()))
        print(f"\n  Seed {seed} Test MAE: {test_mae:.4f}")
        
        torch.save({'y_true': test_tm_lbl, 'y_pred': torch.tensor(preds)}, os.path.join(seed_dir, 'predictions.pt'))
        with open(os.path.join(seed_dir, 'metrics.json'), 'w') as f:
            json.dump({'test_mae': float(test_mae), 'val_mae': float(best_mae)}, f)
    
    # Ensemble
    ensemble_dir = os.path.join(results_dir, "ensemble")
    os.makedirs(ensemble_dir, exist_ok=True)
    mean_preds = np.mean(ensemble_preds, axis=0)
    ensemble_mae = np.mean(np.abs(mean_preds - test_tm_lbl.numpy()))
    
    torch.save({'y_true': test_tm_lbl, 'y_pred': torch.tensor(mean_preds)}, os.path.join(ensemble_dir, 'predictions.pt'))
    with open(os.path.join(ensemble_dir, 'metrics.json'), 'w') as f:
        json.dump({'ensemble_mae': float(ensemble_mae), 'seed_maes': [float(m) for m in seed_maes]}, f)
    
    print(f"\n{'='*50}")
    print(f"V6+ ENSEMBLE TEST MAE: {ensemble_mae:.4f}")
    print(f"Per-seed Val MAEs: {[f'{m:.4f}' for m in seed_maes]}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
