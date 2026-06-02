import os
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

def train_seed(seed, train_ogt_loader, device, save_dir):
    print("\n" + "="*50)
    print(f"Stage 1 OGT Pre-training (Seed {seed})")
    print("="*50)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = StableProtV7(
        emb_dim=CONFIG['input_size'],
        hidden=CONFIG['hidden_size'],
        bottleneck=CONFIG['bottleneck_size'],
        dropout1=CONFIG['dropout_1'],
        dropout2=CONFIG['dropout_2']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['stage1_lr'], weight_decay=CONFIG['weight_decay'])
    criterion = nn.HuberLoss(delta=CONFIG['huber_delta'])
    
    for epoch in range(CONFIG['stage1_epochs']):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_ogt_loader, desc=f"Seed {seed} | Epoch {epoch+1}/{CONFIG['stage1_epochs']}")
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(x, stage='ogt')
            loss = criterion(pred, y)
            loss.backward()
            
            if CONFIG['grad_clip_max_norm'] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip_max_norm'])
                
            optimizer.step()
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
        epoch_loss = train_loss / len(train_ogt_loader)
        print(f"Epoch {epoch+1} finished. Mean Huber Loss: {epoch_loss:.4f}")
        
        # Save model after each epoch
        epoch_model_path = os.path.join(save_dir, f'model_stage1_epoch{epoch+1}.pt')
        torch.save(model.state_dict(), epoch_model_path)
        
    best_model_path = os.path.join(save_dir, 'model_stage1.pt')
    torch.save(model.state_dict(), best_model_path)
    print(f"Saved pre-trained Stage 1 model to {best_model_path}")
    return best_model_path

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/home/bibhu/Documents/temstampto/data/embeddings/prepared_data_v4_cleaned.pt")
    parser.add_argument('--results_dir', type=str, default="results")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = args.data_path
    
    print("Loading dataset...")
    data = torch.load(data_path, map_location="cpu")
    
    train_ogt_emb = data['train_ogt']['embeddings']
    train_ogt_lbl = data['train_ogt']['labels'] if 'labels' in data['train_ogt'] else data['train_ogt']['ogt_original']
    
    print(f"OGT Training data: {train_ogt_emb.shape[0]} samples, dim={train_ogt_emb.shape[1]}")
    
    ogt_loader = DataLoader(
        TmDataset(train_ogt_emb, train_ogt_lbl),
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        drop_last=True
    )
    
    # Resolve relative or absolute path for results_dir
    if not os.path.isabs(args.results_dir):
        results_dir = os.path.join(base_dir, args.results_dir)
    else:
        results_dir = args.results_dir
        
    os.makedirs(results_dir, exist_ok=True)
    
    for seed in CONFIG['seeds']:
        seed_dir = os.path.join(results_dir, f"seed{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        
        train_seed(seed, ogt_loader, device, seed_dir)
        
    print("\nStage 1 OGT Pre-training complete for all seeds.")

if __name__ == "__main__":
    main()
