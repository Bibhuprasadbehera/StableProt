import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from tqdm import tqdm

# Add parent directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from model import StableProtV7
from config import CONFIG

class SimpleDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class WeightedDataset(Dataset):
    def __init__(self, embeddings, labels, weights):
        self.embeddings = embeddings
        self.labels = labels
        self.weights = weights
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx], self.weights[idx]

def compute_inverse_frequency_weights(labels, bin_edges=np.arange(0, 101, 10), min_weight=0.1, max_weight=10.0):
    labels_np = np.array(labels)
    bin_indices = np.digitize(labels_np, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)
    
    counts = np.bincount(bin_indices, minlength=len(bin_edges)-1)
    non_zero_counts = counts[counts > 0]
    median_freq = np.median(non_zero_counts) if len(non_zero_counts) > 0 else 1.0
    
    bin_weights = median_freq / np.maximum(counts, 1)
    sample_weights = bin_weights[bin_indices]
    sample_weights = np.clip(sample_weights, min_weight, max_weight)
    
    return torch.tensor(sample_weights, dtype=torch.float32)

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def train_joint_seed(seed, model_type, train_tm_loader, train_ogt_loader, val_tm_loader, device, save_dir, emb_dim):
    print("\n" + "="*60)
    print(f"Training V7 Joint Multi-Task model (Model: {model_type}, Seed {seed})")
    print("="*60)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = StableProtV7(
        emb_dim=emb_dim,
        use_ogt_feature=False,
        use_tm_feature=False,
        hidden=CONFIG['hidden_size'],
        bottleneck=CONFIG['bottleneck_size']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=CONFIG['lr_scheduler_patience'], factor=CONFIG['lr_scheduler_factor'], min_lr=1e-6
    )
    
    # Use Huber Loss: mean for OGT, none for Tm (to apply weights)
    criterion_ogt = nn.HuberLoss(delta=CONFIG['huber_delta'], reduction='mean')
    criterion_tm = nn.HuberLoss(delta=CONFIG['huber_delta'], reduction='none')
    
    best_val_mae = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(save_dir, 'model_joint.pt')
    
    tm_iter = iter(cycle(train_tm_loader))
    
    # 50 Epochs or early stopping
    num_epochs = 50
    ogt_loss_weight = 0.3
    tm_loss_weight = 1.0
    
    for epoch in range(num_epochs):
        model.train()
        train_ogt_loss = 0.0
        train_tm_loss = 0.0
        
        pbar = tqdm(train_ogt_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (ogt_x, ogt_y) in enumerate(pbar):
            # 1. OGT Forward & Step
            ogt_x, ogt_y = ogt_x.to(device), ogt_y.to(device)
            ogt_pred = model(ogt_x, stage='ogt')
            loss_ogt = criterion_ogt(ogt_pred, ogt_y) * ogt_loss_weight
            
            optimizer.zero_grad()
            loss_ogt.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip_max_norm'])
            optimizer.step()
            train_ogt_loss += loss_ogt.item()
            
            # 2. Tm Forward & Step (with inverse frequency weights)
            tm_x, tm_y, tm_w = next(tm_iter)
            tm_x, tm_y, tm_w = tm_x.to(device), tm_y.to(device), tm_w.to(device)
            tm_pred = model(tm_x, stage='tm')
            loss_tm_elementwise = criterion_tm(tm_pred, tm_y)
            loss_tm = (loss_tm_elementwise * tm_w).mean() * tm_loss_weight
            
            optimizer.zero_grad()
            loss_tm.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip_max_norm'])
            optimizer.step()
            train_tm_loss += loss_tm.item()
            
            if batch_idx % 50 == 0:
                pbar.set_postfix({'OGT Loss': f'{loss_ogt.item():.4f}', 'Tm Loss': f'{loss_tm.item():.4f}'})
                
        # Validation on Tm
        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for x, y in val_tm_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x, stage='tm')
                val_mae += torch.abs(pred - y).sum().item()
        val_mae /= len(val_tm_loader.dataset)
        scheduler.step(val_mae)
        
        print(f"Epoch {epoch+1} - Val MAE: {val_mae:.4f} °C")
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("Saved new best joint model!")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['early_stopping_patience']:
                print(f"Early stopping triggered at epoch {epoch+1}!")
                break
                
    return best_model_path, best_val_mae

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='esm2', choices=['esm2', 'saprot'], help='Model embedding type')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Joint multi-task training on: {device}")
    
    # Setup data paths
    if args.model == 'esm2':
        data_path = os.path.join(SCRIPT_DIR, "../../../../data/embeddings/prepared_data_v4_cleaned.pt")
        emb_dim = 2560
        results_root = os.path.join(SCRIPT_DIR, "results")
    else:
        data_path = os.path.join(SCRIPT_DIR, "../../../../data/embeddings/prepared_data_v4_saprot_cleaned.pt")
        emb_dim = 1280
        results_root = os.path.join(SCRIPT_DIR, "results/saprot")
        
    print(f"Loading cleaned dataset from {data_path}...")
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    
    # Get OGT labels
    train_ogt_lbl = data['train_ogt']['labels'] if 'labels' in data['train_ogt'] else data['train_ogt']['ogt_original']
    train_tm_lbl = data['train_tm']['labels'] if 'labels' in data['train_tm'] else data['train_tm']['tm_consensus']
    val_tm_lbl = data['val_tm']['labels'] if 'labels' in data['val_tm'] else data['val_tm']['tm_consensus']
    test_tm_lbl = data['test_tm']['labels'] if 'labels' in data['test_tm'] else data['test_tm']['tm_consensus']
    
    # Ensure they are FloatTensors
    train_ogt_lbl = torch.tensor(train_ogt_lbl, dtype=torch.float32)
    train_tm_lbl = torch.tensor(train_tm_lbl, dtype=torch.float32)
    val_tm_lbl = torch.tensor(val_tm_lbl, dtype=torch.float32)
    test_tm_lbl = torch.tensor(test_tm_lbl, dtype=torch.float32)
    
    # Create DataLoaders
    train_ogt_loader = DataLoader(
        SimpleDataset(data['train_ogt']['embeddings'], train_ogt_lbl),
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        drop_last=True
    )
    
    # Compute inverse frequency weights for Tm training
    tm_train_weights = compute_inverse_frequency_weights(train_tm_lbl)
    
    train_tm_loader = DataLoader(
        WeightedDataset(data['train_tm']['embeddings'], train_tm_lbl, tm_train_weights),
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        drop_last=True
    )
    val_tm_loader = DataLoader(
        SimpleDataset(data['val_tm']['embeddings'], val_tm_lbl),
        batch_size=CONFIG['batch_size'],
        shuffle=False
    )
    test_tm_loader = DataLoader(
        SimpleDataset(data['test_tm']['embeddings'], test_tm_lbl),
        batch_size=CONFIG['batch_size'],
        shuffle=False
    )
    
    seeds = [1, 2, 3]
    ensemble_preds = []
    
    for seed in seeds:
        seed_dir = os.path.join(results_root, f"seed{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        
        best_model_path, best_mae = train_joint_seed(
            seed, args.model, train_tm_loader, train_ogt_loader, val_tm_loader, device, seed_dir, emb_dim
        )
        
        # Inference on holdout set
        model = StableProtV7(
            emb_dim=emb_dim,
            use_ogt_feature=False,
            use_tm_feature=False,
            hidden=CONFIG['hidden_size'],
            bottleneck=CONFIG['bottleneck_size']
        ).to(device)
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()
        
        preds = []
        with torch.no_grad():
            for x, _ in test_tm_loader:
                p = model(x.to(device), stage='tm').cpu().tolist()
                preds.extend(p)
                
        ensemble_preds.append(preds)
        torch.save(
            {'y_true': test_tm_lbl.numpy(), 'y_pred': np.array(preds)},
            os.path.join(seed_dir, 'predictions_joint.pt')
        )
        print(f"Seed {seed} Joint Model Test MAE: {np.mean(np.abs(np.array(preds) - test_tm_lbl.numpy())):.4f} °C")
        
    # Save ensemble predictions
    ensemble_dir = os.path.join(results_root, "ensemble_joint")
    os.makedirs(ensemble_dir, exist_ok=True)
    mean_preds = np.mean(ensemble_preds, axis=0)
    torch.save(
        {'y_true': test_tm_lbl.numpy(), 'y_pred': mean_preds},
        os.path.join(ensemble_dir, 'predictions_joint.pt')
    )
    
    ensemble_mae = np.mean(np.abs(mean_preds - test_tm_lbl.numpy()))
    print(f"\nFinal V7 Joint ({args.model}) Ensemble Test MAE: {ensemble_mae:.4f} °C")
    
if __name__ == "__main__":
    main()
