import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from config import CONFIG
from model import MultiHead_SaProtPredictor
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

def train_one_seed(seed, train_tm_loader, train_ogt_loader, val_tm_loader, device, save_dir):
    print("\n" + "="*50)
    print(f"Training SaProt Multi-Head Model (Seed {seed})")
    print("="*50)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = MultiHead_SaProtPredictor(
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
    best_model_path = os.path.join(save_dir, 'model.pt')
    
    tm_iter = iter(cycle(train_tm_loader))
    
    for epoch in range(CONFIG['num_epochs']):
        model.train()
        train_ogt_loss = 0.0
        train_tm_loss = 0.0
        
        pbar = tqdm(train_ogt_loader, desc=f"Seed {seed} | Epoch {epoch+1}/{CONFIG['num_epochs']}")
        for batch_idx, (ogt_x, ogt_y) in enumerate(pbar):
            # 1. OGT Forward
            ogt_x, ogt_y = ogt_x.to(device), ogt_y.to(device)
            if CONFIG['mixup_alpha'] > 0:
                ogt_x_mix, ogt_y_mix = mixup_data(ogt_x, ogt_y, CONFIG['mixup_alpha'])
                ogt_pred = model(ogt_x_mix, head='ogt')
                loss_ogt = criterion(ogt_pred, ogt_y_mix).mean() * CONFIG['ogt_loss_weight']
            else:
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
                tm_x_mix, tm_y_mix = mixup_data(tm_x, tm_y, CONFIG['mixup_alpha'])
                tm_pred = model(tm_x_mix, head='tm')
                loss_tm = criterion(tm_pred, tm_y_mix).mean() * CONFIG['tm_loss_weight']
            else:
                tm_pred = model(tm_x, head='tm')
                loss_tm = criterion(tm_pred, tm_y).mean() * CONFIG['tm_loss_weight']
                
            optimizer.zero_grad()
            loss_tm.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip_max_norm'])
            optimizer.step()
            train_tm_loss += loss_tm.item()
            
            if batch_idx % 100 == 0:
                pbar.set_postfix({'OGT': f'{loss_ogt.item():.3f}', 'Tm': f'{loss_tm.item():.3f}'})
            
        # Validation
        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for x, y in val_tm_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x, head='tm')
                val_mae += torch.abs(pred - y).sum().item()
        val_mae /= len(val_tm_loader.dataset)
        scheduler.step(val_mae)
        
        print(f"Epoch {epoch+1} - Val MAE: {val_mae:.4f}")
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("Saved best model!")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['early_stopping_patience']:
                print("Early stopping triggered!")
                break
                
    return best_model_path

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "../../../../data/embeddings/prepared_data_v4_saprot.pt")
    
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
    
    ogt_loader = DataLoader(TmDataset(train_ogt_emb, train_ogt_lbl, is_tm=False), batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    tm_loader = DataLoader(TmDataset(train_tm_emb, train_tm_lbl, is_tm=True), batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(TmDataset(val_tm_emb, val_tm_lbl, is_tm=True), batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(TmDataset(test_tm_emb, test_tm_lbl, is_tm=True), batch_size=CONFIG['batch_size'], shuffle=False)
    
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    ensemble_preds = []
    
    for seed in CONFIG['seeds']:
        seed_dir = os.path.join(results_dir, f"seed{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        
        best_model_path = train_one_seed(seed, tm_loader, ogt_loader, val_loader, device, seed_dir)
        
        # Inference on Test Set
        model = MultiHead_SaProtPredictor(
            hidden1=CONFIG['hidden_size_1'],
            hidden2=CONFIG['hidden_size_2']
        ).to(device)
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        
        preds = []
        with torch.no_grad():
            for x, _ in test_loader:
                p = model(x.to(device), head='tm').cpu()
                preds.extend(p.tolist())
        
        ensemble_preds.append(preds)
        
        torch.save({'y_true': test_tm_lbl, 'y_pred': torch.tensor(preds)}, os.path.join(seed_dir, 'predictions.pt'))
        mae = np.mean(np.abs(np.array(preds) - test_tm_lbl.numpy()))
        with open(os.path.join(seed_dir, 'metrics.json'), 'w') as f:
            json.dump({'mae': float(mae)}, f)
            
    # Ensemble
    ensemble_dir = os.path.join(results_dir, "ensemble")
    os.makedirs(ensemble_dir, exist_ok=True)
    mean_preds = np.mean(ensemble_preds, axis=0)
    mae = np.mean(np.abs(mean_preds - test_tm_lbl.numpy()))
    
    torch.save({'y_true': test_tm_lbl, 'y_pred': torch.tensor(mean_preds)}, os.path.join(ensemble_dir, 'predictions.pt'))
    with open(os.path.join(ensemble_dir, 'metrics.json'), 'w') as f:
        json.dump({'mae': float(mae)}, f)
    print(f"Ensemble SaProt Test MAE: {mae:.4f}")
                
if __name__ == "__main__":
    train()
