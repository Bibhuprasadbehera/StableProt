import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from config import CONFIG
from model import StableProtV7
from tqdm import tqdm

class TmDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

def train_seed_stage2(mode, seed, train_loader, val_loader, test_loader, device, save_dir, stage1_model_path=None):
    print("\n" + "="*50)
    print(f"Stage 2 Tm Training | Mode: {mode} | Seed {seed}")
    print("="*50)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 1. Instantiate the model
    use_ogt_feature = (mode in ['C', 'C2'])
    model = StableProtV7(
        emb_dim=CONFIG['input_size'],
        hidden=CONFIG['hidden_size'],
        bottleneck=CONFIG['bottleneck_size'],
        dropout1=CONFIG['dropout_1'],
        dropout2=CONFIG['dropout_2'],
        use_ogt_feature=use_ogt_feature
    ).to(device)

    # 2. Initialize weights based on the mode
    if mode in ['B', 'C']:
        # Load pre-trained Stage 1 weights
        if stage1_model_path and os.path.exists(stage1_model_path):
            print(f"Loading Stage 1 pre-trained weights from {stage1_model_path}")
            # Load state_dict (which has backbone and ogt_head)
            state_dict = torch.load(stage1_model_path, map_location=device, weights_only=True)
            # Filter out tm_head keys to prevent size mismatch (e.g., when use_ogt_feature=True)
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('tm_head')}
            # Load only the backbone and ogt_head, tm_head remains randomly initialized
            model.load_state_dict(state_dict, strict=False)
        else:
            print(f"WARNING: Stage 1 model path {stage1_model_path} not found. Training from scratch!")
    elif mode == 'C2':
        print("C2 Mode: Randomly initialized backbone.")
        
    # 3. For C and C2, we need the frozen Stage 1 model to predict OGT
    stage1_predictor = None
    if use_ogt_feature:
        if stage1_model_path and os.path.exists(stage1_model_path):
            stage1_predictor = StableProtV7(
                emb_dim=CONFIG['input_size'],
                hidden=CONFIG['hidden_size'],
                bottleneck=CONFIG['bottleneck_size']
            ).to(device)
            stage1_predictor.load_state_dict(torch.load(stage1_model_path, map_location=device, weights_only=True))
            stage1_predictor.eval()
            for p in stage1_predictor.parameters():
                p.requires_grad = False
        else:
            print("ERROR: Stage 1 pre-trained model required for predicting OGT feature in C/C2 modes.")
            return None

    # 4. Set up optimizers and learning rates
    if mode == 'C2':
        # No pre-trained weights to preserve, train everything with head_lr
        optimizer = optim.Adam(
            model.parameters(),
            lr=CONFIG['stage2_head_lr'],
            weight_decay=CONFIG['weight_decay']
        )
    else:
        # Differential learning rates for backbone vs head
        param_groups = model.get_stage2_param_groups(
            backbone_lr=CONFIG['stage2_backbone_lr'],
            head_lr=CONFIG['stage2_head_lr']
        )
        optimizer = optim.Adam(param_groups, weight_decay=CONFIG['weight_decay'])
        
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=CONFIG['lr_scheduler_patience'],
        factor=CONFIG['lr_scheduler_factor'],
        min_lr=1e-6
    )
    
    criterion = nn.HuberLoss(delta=CONFIG['huber_delta'])
    
    best_val_mae = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(save_dir, f'model_{mode}.pt')
    
    for epoch in range(CONFIG['stage2_epochs']):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['stage2_epochs']}")
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            
            # Predict OGT if needed
            ogt_pred = None
            if use_ogt_feature:
                with torch.no_grad():
                    ogt_pred = stage1_predictor(x, stage='ogt')
            
            optimizer.zero_grad()
            pred = model(x, stage='tm', ogt_pred=ogt_pred)
            loss = criterion(pred, y)
            loss.backward()
            
            if CONFIG['grad_clip_max_norm'] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip_max_norm'])
                
            optimizer.step()
            train_loss += loss.item()
            
            if batch_idx % 20 == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
        # Validation
        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                
                ogt_pred = None
                if use_ogt_feature:
                    ogt_pred = stage1_predictor(x, stage='ogt')
                    
                pred = model(x, stage='tm', ogt_pred=ogt_pred)
                val_mae += torch.abs(pred - y).sum().item()
                
        val_mae /= len(val_loader.dataset)
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
                
    # Load best model for testing
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    preds = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            ogt_pred = None
            if use_ogt_feature:
                ogt_pred = stage1_predictor(x, stage='ogt')
                
            p = model(x, stage='tm', ogt_pred=ogt_pred).cpu()
            preds.extend(p.tolist())
            
    # Save predictions
    test_labels = torch.cat([y for _, y in test_loader])
    torch.save(
        {'y_true': test_labels, 'y_pred': torch.tensor(preds)},
        os.path.join(save_dir, f'predictions_{mode}.pt')
    )
    
    mae = np.mean(np.abs(np.array(preds) - test_labels.numpy()))
    print(f"Mode {mode} | Seed {seed} | Test MAE: {mae:.4f}")
    
    with open(os.path.join(save_dir, f'metrics_{mode}.json'), 'w') as f:
        json.dump({'mae': float(mae)}, f)
        
    return preds

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['B', 'C', 'C2'])
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = "/home/bibhu/Documents/temstampto/data/embeddings/prepared_data_v4_cleaned.pt"
    
    print("Loading dataset...")
    data = torch.load(data_path, map_location="cpu")
    
    train_tm_emb = data['train_tm']['embeddings']
    train_tm_lbl = data['train_tm']['labels']
    val_tm_emb = data['val_tm']['embeddings']
    val_tm_lbl = data['val_tm']['labels']
    test_tm_emb = data['test_tm']['embeddings']
    test_tm_lbl = data['test_tm']['labels']
    
    train_loader = DataLoader(TmDataset(train_tm_emb, train_tm_lbl), batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(TmDataset(val_tm_emb, val_tm_lbl), batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(TmDataset(test_tm_emb, test_tm_lbl), batch_size=CONFIG['batch_size'], shuffle=False)
    
    results_dir = os.path.join(base_dir, 'results')
    
    ensemble_preds = []
    
    for seed in CONFIG['seeds']:
        seed_dir = os.path.join(results_dir, f"seed{seed}")
        stage1_model_path = os.path.join(seed_dir, 'model_stage1.pt')
        
        preds = train_seed_stage2(
            args.mode, seed, train_loader, val_loader, test_loader,
            device, seed_dir, stage1_model_path=stage1_model_path
        )
        if preds is not None:
            ensemble_preds.append(preds)
            
    # Ensemble predictions
    if ensemble_preds:
        ensemble_dir = os.path.join(results_dir, f"ensemble_{args.mode}")
        os.makedirs(ensemble_dir, exist_ok=True)
        mean_preds = np.mean(ensemble_preds, axis=0)
        mae = np.mean(np.abs(mean_preds - test_tm_lbl.numpy()))
        
        torch.save(
            {'y_true': test_tm_lbl, 'y_pred': torch.tensor(mean_preds)},
            os.path.join(ensemble_dir, 'predictions.pt')
        )
        
        with open(os.path.join(ensemble_dir, 'metrics.json'), 'w') as f:
            json.dump({'mae': float(mae)}, f)
            
        print(f"\n[Ensemble Mode {args.mode}] Test MAE: {mae:.4f}")

if __name__ == "__main__":
    main()
