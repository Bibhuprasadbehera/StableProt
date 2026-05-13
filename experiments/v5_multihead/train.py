import os
import json
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

def train_one_seed(seed, train_tm_loader, train_ogt_loader, val_tm_loader, tm_mean, tm_std, ogt_mean, ogt_std, device, save_dir):
    print("\n" + "="*50)
    print(f"Training Multi-Head Model (Seed {seed})")
    print("="*50)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
    
    criterion = nn.HuberLoss(delta=CONFIG['huber_delta'])
    
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
                loss_ogt = criterion(ogt_pred, ogt_y_mix) * CONFIG['ogt_loss_weight']
            else:
                ogt_pred = model(ogt_x, head='ogt')
                loss_ogt = criterion(ogt_pred, ogt_y) * CONFIG['ogt_loss_weight']
            
            if torch.isnan(loss_ogt):
                print(f"NaN in OGT loss at batch {batch_idx}! OGT pred range: {ogt_pred.min().item():.3f}-{ogt_pred.max().item():.3f}")
                break
            
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
                loss_tm = criterion(tm_pred, tm_y_mix) * CONFIG['tm_loss_weight']
            else:
                tm_pred = model(tm_x, head='tm')
                loss_tm = criterion(tm_pred, tm_y) * CONFIG['tm_loss_weight']
            
            if torch.isnan(loss_tm):
                print(f"NaN in Tm loss at batch {batch_idx}! Tm pred range: {tm_pred.min().item():.3f}-{tm_pred.max().item():.3f}")
                break
                
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
                if CONFIG['target_normalization']:
                    pred = pred * tm_std + tm_mean
                    y = y * tm_std + tm_mean
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
    data_path = os.path.join(base_dir, "../../new_data/prepared_data_v5_prott5.pt")
    
    print("Loading dataset...")
    data = torch.load(data_path, weights_only=True)
    
    train_ogt_emb = data['train_ogt']['embeddings']
    train_ogt_lbl = data['train_ogt']['labels']
    train_tm_emb = data['train_tm']['embeddings']
    train_tm_lbl = data['train_tm']['labels']
    val_tm_emb = data['val_tm']['embeddings']
    val_tm_lbl = data['val_tm']['labels']
    test_tm_emb = data['test_tm']['embeddings']
    test_tm_lbl = data['test_tm']['labels']
    
    # Normalization
    tm_mean, tm_std = train_tm_lbl.mean().item(), train_tm_lbl.std().item()
    ogt_mean, ogt_std = train_ogt_lbl.mean().item(), train_ogt_lbl.std().item()
    
    if CONFIG['target_normalization']:
        train_tm_norm = (train_tm_lbl - tm_mean) / tm_std
        val_tm_norm = (val_tm_lbl - tm_mean) / tm_std
        train_ogt_norm = (train_ogt_lbl - ogt_mean) / ogt_std
    else:
        train_tm_norm, val_tm_norm, train_ogt_norm = train_tm_lbl, val_tm_lbl, train_ogt_lbl
        
    ogt_loader = DataLoader(TmDataset(train_ogt_emb, train_ogt_norm, is_tm=False), batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    tm_loader = DataLoader(TmDataset(train_tm_emb, train_tm_norm, is_tm=True), batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(TmDataset(val_tm_emb, val_tm_norm, is_tm=True), batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(TmDataset(test_tm_emb, test_tm_lbl, is_tm=True), batch_size=CONFIG['batch_size'], shuffle=False)
    
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    ensemble_preds = []
    
    for seed in CONFIG['seeds']:
        seed_dir = os.path.join(results_dir, f"seed{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        
        best_model_path = train_one_seed(seed, tm_loader, ogt_loader, val_loader, tm_mean, tm_std, ogt_mean, ogt_std, device, seed_dir)
        
        # Inference on Test Set
        model = MultiHead_TmPredictor(
            input_size=CONFIG['input_size'],
            hidden1=CONFIG['hidden_size_1'],
            hidden2=CONFIG['hidden_size_2']
        ).to(device)
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        
        preds = []
        with torch.no_grad():
            for x, _ in test_loader:
                p = model(x.to(device), head='tm').cpu()
                if CONFIG['target_normalization']:
                    p = p * tm_std + tm_mean
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
    print(f"Ensemble Tm Test MAE: {mae:.4f}")
    
    # ── Also evaluate OGT head on 210K OGT test set ──
    print("\nEvaluating OGT head on 210K test set...")
    test_ogt_emb = data['test_ogt']['embeddings']
    test_ogt_lbl = data['test_ogt']['labels']
    ogt_test_loader = DataLoader(TmDataset(test_ogt_emb, test_ogt_lbl, is_tm=False), batch_size=CONFIG['batch_size'], shuffle=False)
    
    ogt_ensemble_preds = []
    for seed in CONFIG['seeds']:
        seed_dir = os.path.join(results_dir, f"seed{seed}")
        model = MultiHead_TmPredictor(input_size=CONFIG['input_size'], hidden1=CONFIG['hidden_size_1'], hidden2=CONFIG['hidden_size_2']).to(device)
        model.load_state_dict(torch.load(os.path.join(seed_dir, 'model.pt')))
        model.eval()
        preds = []
        with torch.no_grad():
            for x, _ in ogt_test_loader:
                p = model(x.to(device), head='ogt').cpu()
                if CONFIG['target_normalization']:
                    p = p * ogt_std + ogt_mean
                preds.extend(p.tolist())
        ogt_ensemble_preds.append(preds)
    
    ogt_mean_preds = np.mean(ogt_ensemble_preds, axis=0)
    ogt_mae = np.mean(np.abs(ogt_mean_preds - test_ogt_lbl.numpy()))
    
    ogt_results_dir = os.path.join(ensemble_dir, 'ogt_eval')
    os.makedirs(ogt_results_dir, exist_ok=True)
    torch.save({'y_true': test_ogt_lbl, 'y_pred': torch.tensor(ogt_mean_preds)}, os.path.join(ogt_results_dir, 'predictions.pt'))
    with open(os.path.join(ogt_results_dir, 'metrics.json'), 'w') as f:
        json.dump({'mae': float(ogt_mae)}, f)
    print(f"OGT Head Ensemble MAE on 210K test: {ogt_mae:.4f}")
                
if __name__ == "__main__":
    train()
