import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from config import CONFIG
from model import MultiHead_TmPredictor
from tqdm import tqdm

class TmDataset(Dataset):
    def __init__(self, embeddings, labels, is_tm=True):
        self.embeddings = embeddings
        self.labels = labels
        self.is_tm = is_tm
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

def get_binned_weights(labels, num_bins=8, min_val=0.0, max_val=100.0):
    labels_np = labels.numpy()
    bins = np.linspace(min_val, max_val, num_bins + 1)
    inds = np.digitize(labels_np, bins) - 1
    inds = np.clip(inds, 0, num_bins - 1)
    
    counts = np.bincount(inds, minlength=num_bins)
    counts = np.where(counts == 0, 1, counts)
    
    total = len(labels_np)
    weights_per_bin = total / (num_bins * counts)
    sample_weights = weights_per_bin[inds]
    return torch.tensor(sample_weights, dtype=torch.float32)

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_path = "../prepared_data_v2.pt"
    if not os.path.exists(data_path):
        print(f"Dataset {data_path} not found! Please run prepare_data_v2.py first.")
        return
        
    print("Loading dataset...")
    data = torch.load(data_path, weights_only=True)
    
    train_ogt_emb = data['train_ogt']['embeddings']
    train_ogt_lbl = data['train_ogt']['labels']
    
    train_tm_emb = data['train_tm']['embeddings']
    train_tm_lbl = data['train_tm']['labels']
    
    val_tm_emb = data['val_tm']['embeddings']
    val_tm_lbl = data['val_tm']['labels']
    
    print(f"OGT Train: {len(train_ogt_lbl)}, Tm Train: {len(train_tm_lbl)}, Tm Val: {len(val_tm_lbl)}")
    
    # Normalization parameters for Tm
    tm_mean = train_tm_lbl.mean().item()
    tm_std = train_tm_lbl.std().item()
    ogt_mean = train_ogt_lbl.mean().item()
    ogt_std = train_ogt_lbl.std().item()
    print(f"Tm normalization - Mean: {tm_mean:.2f}, Std: {tm_std:.2f}")
    print(f"OGT normalization - Mean: {ogt_mean:.2f}, Std: {ogt_std:.2f}")
    
    if CONFIG['target_normalization']:
        train_tm_lbl = (train_tm_lbl - tm_mean) / tm_std
        val_tm_lbl = (val_tm_lbl - tm_mean) / tm_std
        train_ogt_lbl = (train_ogt_lbl - ogt_mean) / ogt_std
    
    # Datasets
    ogt_dataset = TmDataset(train_ogt_emb, train_ogt_lbl, is_tm=False)
    tm_dataset = TmDataset(train_tm_emb, train_tm_lbl, is_tm=True)
    val_dataset = TmDataset(val_tm_emb, val_tm_lbl, is_tm=True)
    
    # Dataloaders
    ogt_loader = DataLoader(ogt_dataset, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    tm_loader = DataLoader(tm_dataset, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    model = MultiHead_TmPredictor(
        input_size=CONFIG['input_size'],
        hidden1=CONFIG['hidden_size_1'],
        hidden2=CONFIG['hidden_size_2'],
        dropout1=CONFIG['dropout_1'],
        dropout2=CONFIG['dropout_2']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=CONFIG['lr_scheduler_patience'], factor=CONFIG['lr_scheduler_factor'], min_lr=1e-6
    )
    
    criterion = nn.HuberLoss(delta=CONFIG['huber_delta'], reduction='none')
    
    best_val_mae = float('inf')
    patience_counter = 0
    
    # Per-sample weights
    ogt_weights = get_binned_weights(data['train_ogt']['labels']).to(device)
    tm_weights = get_binned_weights(data['train_tm']['labels']).to(device)
    
    for epoch in range(CONFIG['num_epochs']):
        model.train()
        train_ogt_loss = 0.0
        train_tm_loss = 0.0
        
        # We need an iterator for tm_loader to cycle through it since it's smaller than ogt_loader
        def cycle(iterable):
            while True:
                for x in iterable:
                    yield x
                    
        tm_iter = iter(cycle(tm_loader))
        
        pbar = tqdm(ogt_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        for ogt_x, ogt_y in pbar:
            # 1. OGT Forward
            ogt_x, ogt_y = ogt_x.to(device), ogt_y.to(device)
            # Mixup
            if CONFIG['mixup_alpha'] > 0:
                ogt_x, ogt_y = mixup_data(ogt_x, ogt_y, CONFIG['mixup_alpha'])
                
            ogt_pred = model(ogt_x, head='ogt')
            loss_ogt = criterion(ogt_pred, ogt_y).mean() * CONFIG['ogt_loss_weight']
            
            optimizer.zero_grad()
            loss_ogt.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip_max_norm'])
            optimizer.step()
            train_ogt_loss += loss_ogt.item()
            
            # 2. Tm Forward
            tm_x, tm_y = next(tm_iter)
            tm_x, tm_y = tm_x.to(device), tm_y.to(device)
            
            if CONFIG['mixup_alpha'] > 0:
                tm_x, tm_y = mixup_data(tm_x, tm_y, CONFIG['mixup_alpha'])
                
            tm_pred = model(tm_x, head='tm')
            loss_tm = criterion(tm_pred, tm_y).mean() * CONFIG['tm_loss_weight']
            
            optimizer.zero_grad()
            loss_tm.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip_max_norm'])
            optimizer.step()
            train_tm_loss += loss_tm.item()
            
            pbar.set_postfix({'OGT Loss': f'{loss_ogt.item():.3f}', 'Tm Loss': f'{loss_tm.item():.3f}'})
            
        # Validation
        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x, head='tm')
                
                if CONFIG['target_normalization']:
                    pred = pred * tm_std + tm_mean
                    y = y * tm_std + tm_mean
                    
                val_mae += torch.abs(pred - y).sum().item()
                
        val_mae /= len(val_dataset)
        scheduler.step(val_mae)
        
        print(f"Epoch {epoch+1} - Val MAE: {val_mae:.4f}")
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), 'results/best_model.pth')
            print("Saved best model!")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['early_stopping_patience']:
                print("Early stopping triggered!")
                break
                
if __name__ == "__main__":
    train()
