"""StableProt V8 Disjoint Multi-Head Predictor with Enriched Auxiliary Features.

Implements disjoint MLP backbones for Tm and OGT prediction with:
1. 1289-dim / 1288-dim enriched inputs (SaProt + OGT prior + TM flag + length + 6-dim AA composition)
2. Heteroscedastic Gaussian NLL uncertainty head for Tm (predicting mean and log-variance)
3. Target z-score standardization and inverse frequency temperature bin weighting
4. Data sanitization (purging unphysical Tm < OGT records and short peptides <50 AA)
5. True alternating optimization loop with AMP mixed precision and --debug mode
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

CONFIG = {
    'input_size_tm': 1289,
    'input_size_ogt': 1288,
    'hidden_size_1': 512,
    'hidden_size_2': 256,
    'dropout_1': 0.3,
    'dropout_2': 0.2,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'batch_size': 64,
    'max_epochs': 100,
    'early_stopping_patience': 10,
    'huber_delta': 5.0,
    'grad_clip_max_norm': 1.0,
    'seeds': [1, 2, 3, 4, 5],
    'bin_edges': list(range(0, 106, 5)),
    'weight_clamp_min': 0.5,
    'weight_clamp_max': 5.0,
    'target_jitter_std': 0.3,
}

def compute_aa_composition(sequences):
    """Compute 6-dim bounded AA composition vector in [0, 1]."""
    comp_list = []
    for seq in sequences:
        s = str(seq).upper()
        # If sequence has 3Di tokens (e.g. Foldseek MaEpKdLq), clean out lowercase 3Di characters
        s_clean = "".join([c for c in s if c.isupper() and c.isalpha()])
        L = max(len(s_clean), 1)
        
        f_charged = sum(s_clean.count(aa) for aa in "DEKR") / L
        f_hydro = sum(s_clean.count(aa) for aa in "VILMFWC") / L
        f_proline = s_clean.count("P") / L
        f_cysteine = s_clean.count("C") / L
        f_aromatic = sum(s_clean.count(aa) for aa in "FWY") / L
        
        # Aliphatic index formula normalized to [0, 1]
        a = s_clean.count("A")
        v = s_clean.count("V")
        il = s_clean.count("I") + s_clean.count("L")
        aliph_raw = (a + 2.9 * v + 3.9 * il) / (L * 100.0)
        aliph_norm = min(aliph_raw, 1.0)
        
        comp_list.append([
            min(f_charged, 1.0),
            min(f_hydro, 1.0),
            min(f_proline, 1.0),
            min(f_cysteine, 1.0),
            min(f_aromatic, 1.0),
            aliph_norm
        ])
    return torch.tensor(comp_list, dtype=torch.float32)

def enrich_inputs(embeddings, sequences, tmhmm_flags=None, ogt_priors=None):
    """Construct 1289-dim (if ogt_priors provided) or 1288-dim feature tensor."""
    N = embeddings.shape[0]
    lengths = []
    for seq in sequences:
        s = str(seq).upper()
        s_clean = "".join([c for c in s if c.isupper() and c.isalpha()])
        lengths.append(min(len(s_clean) / 2048.0, 1.0))
    lengths_t = torch.tensor(lengths, dtype=torch.float32).unsqueeze(-1)
    
    if tmhmm_flags is not None:
        tm_t = torch.tensor([float(f) for f in tmhmm_flags], dtype=torch.float32).unsqueeze(-1)
    else:
        tm_t = torch.zeros((N, 1), dtype=torch.float32)
        
    aa_comp = compute_aa_composition(sequences)
    
    if ogt_priors is not None:
        ogt_t = torch.tensor([float(o) / 100.0 for o in ogt_priors], dtype=torch.float32).unsqueeze(-1)
        return torch.cat([embeddings.float(), ogt_t, tm_t, lengths_t, aa_comp], dim=-1)
    else:
        return torch.cat([embeddings.float(), tm_t, lengths_t, aa_comp], dim=-1)

class MultiHeadSaProtV8(nn.Module):
    def __init__(self, input_dim_tm=1289, input_dim_ogt=1288, hidden1=512, hidden2=256, dropout1=0.3, dropout2=0.2):
        super().__init__()
        # Disjoint Tm pathway
        self.fc1_tm = nn.Linear(input_dim_tm, hidden1)
        self.ln1_tm = nn.LayerNorm(hidden1)
        self.fc2_tm = nn.Linear(hidden1, hidden2)
        self.ln2_tm = nn.LayerNorm(hidden2)
        self.res_tm = nn.Linear(hidden1, hidden2)
        self.head_tm = nn.Linear(hidden2, 2)  # [z_mean, log_z_var]
        
        # Disjoint OGT pathway
        self.fc1_ogt = nn.Linear(input_dim_ogt, hidden1)
        self.ln1_ogt = nn.LayerNorm(hidden1)
        self.fc2_ogt = nn.Linear(hidden1, hidden2)
        self.ln2_ogt = nn.LayerNorm(hidden2)
        self.res_ogt = nn.Linear(hidden1, hidden2)
        self.head_ogt = nn.Linear(hidden2, 1)
        
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout1)
        self.drop2 = nn.Dropout(dropout2)
        
    def forward(self, x, head='tm'):
        if head == 'tm':
            h1 = self.drop1(self.act(self.ln1_tm(self.fc1_tm(x))))
            h2 = self.drop2(self.act(self.ln2_tm(self.fc2_tm(h1)) + self.res_tm(h1)))
            out = self.head_tm(h2)
            return out[:, 0], out[:, 1]  # z_mean, log_z_var
        else:
            h1 = self.drop1(self.act(self.ln1_ogt(self.fc1_ogt(x))))
            h2 = self.drop2(self.act(self.ln2_ogt(self.fc2_ogt(h1)) + self.res_ogt(h1)))
            return self.head_ogt(h2).squeeze(-1)

class ProteinDataset(Dataset):
    def __init__(self, x, y, weights=None):
        self.x = x
        self.y = y
        self.weights = weights if weights is not None else torch.ones_like(y)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.weights[idx]

def compute_bin_weights(labels, bin_edges, clamp_min=0.5, clamp_max=5.0):
    labels_np = labels.numpy() if hasattr(labels, 'numpy') else np.array(labels)
    bin_idx = np.digitize(labels_np, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, len(bin_edges) - 2)
    counts = np.bincount(bin_idx, minlength=len(bin_edges)-1).astype(float)
    counts[counts == 0] = 1.0
    med = np.median(counts[counts > 0])
    w = np.sqrt(med / counts)
    w = np.clip(w, clamp_min, clamp_max)
    return torch.tensor(w[bin_idx], dtype=torch.float32)

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def sanitize_data(dict_data, is_tm=True):
    """Purge Tm < OGT outliers and <50 AA sequences."""
    seqs = dict_data['sequences']
    embs = dict_data['embeddings']
    if is_tm:
        lbls = dict_data.get('tm_consensus', dict_data.get('labels'))
        ogts = dict_data.get('ogt', [50.0]*len(seqs))
    else:
        lbls = dict_data.get('ogt_consensus', dict_data.get('labels'))
        ogts = lbls
        
    keep_indices = []
    for idx, seq in enumerate(seqs):
        s_clean = "".join([c for c in str(seq).upper() if c.isupper() and c.isalpha()])
        if len(s_clean) < 50:
            continue
        if is_tm and ogts is not None and idx < len(ogts):
            o_val = float(ogts[idx]) if ogts[idx] is not None else 0.0
            if float(lbls[idx]) < o_val:
                continue
        keep_indices.append(idx)
        
    print(f"  Sanitization: kept {len(keep_indices)} / {len(seqs)} sequences ({len(seqs)-len(keep_indices)} purged).")
    embs_sub = embs[keep_indices]
    seqs_sub = [seqs[i] for i in keep_indices]
    lbls_sub = torch.tensor([float(lbls[i]) for i in keep_indices], dtype=torch.float32)
    
    if is_tm:
        ogts_sub = [ogts[i] for i in keep_indices]
        tmhmm_sub = [dict_data.get('tmhmm_tm_binary', [0]*len(seqs))[i] for i in keep_indices]
        return embs_sub, seqs_sub, lbls_sub, ogts_sub, tmhmm_sub
    else:
        tmhmm_sub = [dict_data.get('tmhmm_tm_binary', [0]*len(seqs))[i] for i in keep_indices]
        return embs_sub, seqs_sub, lbls_sub, tmhmm_sub

def train_seed(seed, train_tm_ds, train_ogt_ds, val_tm_ds, val_ogt_ds, tm_mean, tm_std, device, save_dir, max_epochs, patience):
    print(f"\n{'='*50}\nTraining V8 Disjoint Ensemble (Seed {seed})\n{'='*50}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    model = MultiHeadSaProtV8(
        input_dim_tm=CONFIG['input_size_tm'],
        input_dim_ogt=CONFIG['input_size_ogt'],
        hidden1=CONFIG['hidden_size_1'],
        hidden2=CONFIG['hidden_size_2'],
        dropout1=CONFIG['dropout_1'],
        dropout2=CONFIG['dropout_2']
    ).to(device)
    
    optimizer = optim.AdamW([
        {'params': model.fc1_tm.parameters(), 'lr': CONFIG['learning_rate']},
        {'params': model.ln1_tm.parameters(), 'lr': CONFIG['learning_rate']},
        {'params': model.fc2_tm.parameters(), 'lr': CONFIG['learning_rate']},
        {'params': model.ln2_tm.parameters(), 'lr': CONFIG['learning_rate']},
        {'params': model.res_tm.parameters(), 'lr': CONFIG['learning_rate']},
        {'params': model.head_tm.parameters(), 'lr': CONFIG['learning_rate']},
        {'params': model.fc1_ogt.parameters(), 'lr': CONFIG['learning_rate']},
        {'params': model.ln1_ogt.parameters(), 'lr': CONFIG['learning_rate']},
        {'params': model.fc2_ogt.parameters(), 'lr': CONFIG['learning_rate']},
        {'params': model.ln2_ogt.parameters(), 'lr': CONFIG['learning_rate']},
        {'params': model.res_ogt.parameters(), 'lr': CONFIG['learning_rate']},
        {'params': model.head_ogt.parameters(), 'lr': CONFIG['learning_rate']},
    ], weight_decay=CONFIG['weight_decay'])
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-6)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    tm_loader = DataLoader(train_tm_ds, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    ogt_loader = DataLoader(train_ogt_ds, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    val_tm_loader = DataLoader(val_tm_ds, batch_size=CONFIG['batch_size'], shuffle=False)
    val_ogt_loader = DataLoader(val_ogt_ds, batch_size=CONFIG['batch_size'], shuffle=False)
    
    tm_iter = iter(cycle(tm_loader))
    best_val_mae = float('inf')
    patience_cnt = 0
    best_path = os.path.join(save_dir, 'model.pt')
    
    for epoch in range(max_epochs):
        model.train()
        train_tm_loss = 0.0
        train_ogt_loss = 0.0
        
        pbar = tqdm(ogt_loader, desc=f"Seed {seed} | Ep {epoch+1}/{max_epochs}", leave=False)
        for x_ogt, y_ogt, w_ogt in pbar:
            # 1. OGT Step (Huber on full 941k dataset)
            x_ogt, y_ogt, w_ogt = x_ogt.to(device), y_ogt.to(device), w_ogt.to(device)
            if CONFIG['target_jitter_std'] > 0:
                y_ogt = y_ogt + torch.randn_like(y_ogt) * CONFIG['target_jitter_std']
                
            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast('cuda'):
                    pred_ogt = model(x_ogt, head='ogt')
                    huber = nn.functional.huber_loss(pred_ogt, y_ogt, delta=CONFIG['huber_delta'], reduction='none')
                    loss_ogt = (huber * w_ogt).mean()
                scaler.scale(loss_ogt).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip_max_norm'])
                if not torch.isnan(loss_ogt):
                    scaler.step(optimizer)
                    scaler.update()
            else:
                pred_ogt = model(x_ogt, head='ogt')
                huber = nn.functional.huber_loss(pred_ogt, y_ogt, delta=CONFIG['huber_delta'], reduction='none')
                loss_ogt = (huber * w_ogt).mean()
                loss_ogt.backward()
                nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip_max_norm'])
                if not torch.isnan(loss_ogt):
                    optimizer.step()
            train_ogt_loss += loss_ogt.item()

            # 2. Tm Step (Gaussian NLL on cycled iterator)
            x_tm, y_tm, w_tm = next(tm_iter)
            x_tm, y_tm, w_tm = x_tm.to(device), y_tm.to(device), w_tm.to(device)
            if CONFIG['target_jitter_std'] > 0:
                y_tm = y_tm + torch.randn_like(y_tm) * CONFIG['target_jitter_std']
                
            z_tm = (y_tm - tm_mean) / tm_std
            
            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast('cuda'):
                    z_mu, log_z_var = model(x_tm, head='tm')
                    nll = 0.5 * torch.exp(-log_z_var) * (z_mu - z_tm)**2 + 0.5 * log_z_var
                    loss_tm = (nll * w_tm).mean()
                scaler.scale(loss_tm).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip_max_norm'])
                if not torch.isnan(loss_tm):
                    scaler.step(optimizer)
                    scaler.update()
            else:
                z_mu, log_z_var = model(x_tm, head='tm')
                nll = 0.5 * torch.exp(-log_z_var) * (z_mu - z_tm)**2 + 0.5 * log_z_var
                loss_tm = (nll * w_tm).mean()
                loss_tm.backward()
                nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip_max_norm'])
                if not torch.isnan(loss_tm):
                    optimizer.step()
            train_tm_loss += loss_tm.item()
            
        # Validation
        model.eval()
        val_tm_mae = 0.0
        with torch.no_grad():
            for x, y, _ in val_tm_loader:
                x, y = x.to(device), y.to(device)
                z_mu, _ = model(x, head='tm')
                pred_y = z_mu * tm_std + tm_mean
                val_tm_mae += torch.abs(pred_y - y).sum().item()
        val_tm_mae /= len(val_tm_loader.dataset)
        
        val_ogt_mae = 0.0
        with torch.no_grad():
            for x, y, _ in val_ogt_loader:
                x, y = x.to(device), y.to(device)
                pred_o = model(x, head='ogt')
                val_ogt_mae += torch.abs(pred_o - y).sum().item()
        val_ogt_mae /= len(val_ogt_loader.dataset)
        
        if epoch >= 5:
            scheduler.step(val_tm_mae)
            
        print(f"  Ep {epoch+1:3d} | Val Tm MAE: {val_tm_mae:.4f}°C | Val OGT MAE: {val_ogt_mae:.4f}°C")
        
        if val_tm_mae < best_val_mae:
            best_val_mae = val_tm_mae
            patience_cnt = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  Early stopping triggered after {epoch+1} epochs.")
                break
                
    return best_path, best_val_mae

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Run quick 5-epoch debug test on 1% subset")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Debug mode: {args.debug}")
    
    data_path = "data/embeddings/saprot_tm_struct_embeddings.pt"
    print(f"Loading master dataset from {data_path}...")
    data = torch.load(data_path, map_location='cpu')
    
    if args.debug:
        print("Debug mode enabled: slicing dictionaries to 1% subset before sanitization...")
        for k in ['embeddings', 'sequences', 'tm_consensus', 'ogt', 'tmhmm_tm_binary', 'ids']:
            if k in data['train_tm']:
                data['train_tm'][k] = data['train_tm'][k][:200]
            if k in data['val_tm']:
                data['val_tm'][k] = data['val_tm'][k][:50]
        for k in ['embeddings', 'sequences', 'ogt_consensus', 'ogt_original', 'tmhmm_tm_binary', 'ids']:
            if k in data['train_ogt']:
                data['train_ogt'][k] = data['train_ogt'][k][:1000]

    print("Sanitizing datasets...")
    tr_tm_emb, tr_tm_seq, tr_tm_lbl, tr_tm_ogt, tr_tm_tmhmm = sanitize_data(data['train_tm'], is_tm=True)
    val_tm_emb, val_tm_seq, val_tm_lbl, val_tm_ogt, val_tm_tmhmm = sanitize_data(data['val_tm'], is_tm=True)
    tr_ogt_emb, tr_ogt_seq, tr_ogt_lbl, tr_ogt_tmhmm = sanitize_data(data['train_ogt'], is_tm=False)
    
    val_ogt_emb = tr_ogt_emb[:min(2000, len(tr_ogt_emb))]
    val_ogt_seq = tr_ogt_seq[:min(2000, len(tr_ogt_seq))]
    val_ogt_lbl = tr_ogt_lbl[:min(2000, len(tr_ogt_lbl))]
    val_ogt_tmhmm = tr_ogt_tmhmm[:min(2000, len(tr_ogt_tmhmm))]
    
    if args.debug:
        seeds = [1]
        max_epochs = 5
        patience = 5
    else:
        seeds = CONFIG['seeds']
        max_epochs = CONFIG['max_epochs']
        patience = CONFIG['early_stopping_patience']
        
    tm_mean = tr_tm_lbl.mean().item()
    tm_std = tr_tm_lbl.std().item()
    print(f"Target Tm Z-score scaling | Mean: {tm_mean:.2f}°C, Std: {tm_std:.2f}°C")
    
    print("Enriching features (1289-dim Tm / 1288-dim OGT)...")
    tr_tm_x = enrich_inputs(tr_tm_emb, tr_tm_seq, tr_tm_tmhmm, tr_tm_ogt)
    val_tm_x = enrich_inputs(val_tm_emb, val_tm_seq, val_tm_tmhmm, val_tm_ogt)
    tr_ogt_x = enrich_inputs(tr_ogt_emb, tr_ogt_seq, tr_ogt_tmhmm)
    val_ogt_x = enrich_inputs(val_ogt_emb, val_ogt_seq, val_ogt_tmhmm)
    
    tr_tm_w = compute_bin_weights(tr_tm_lbl, CONFIG['bin_edges'])
    val_tm_w = torch.ones_like(val_tm_lbl)
    tr_ogt_w = compute_bin_weights(tr_ogt_lbl, CONFIG['bin_edges'])
    val_ogt_w = torch.ones_like(val_ogt_lbl)
    
    train_tm_ds = ProteinDataset(tr_tm_x, tr_tm_lbl, tr_tm_w)
    val_tm_ds = ProteinDataset(val_tm_x, val_tm_lbl, val_tm_w)
    train_ogt_ds = ProteinDataset(tr_ogt_x, tr_ogt_lbl, tr_ogt_w)
    val_ogt_ds = ProteinDataset(val_ogt_x, val_ogt_lbl, val_ogt_w)
    
    save_dir = "experiments/src/training/v8_disjoint/results"
    os.makedirs(save_dir, exist_ok=True)
    
    seed_maes = []
    for s in seeds:
        s_dir = os.path.join(save_dir, f"seed{s}")
        os.makedirs(s_dir, exist_ok=True)
        _, best_mae = train_seed(s, train_tm_ds, train_ogt_ds, val_tm_ds, val_ogt_ds, tm_mean, tm_std, device, s_dir, max_epochs, patience)
        seed_maes.append(best_mae)
        
    print(f"\nTraining Complete | Mean Val MAE across seeds: {np.mean(seed_maes):.4f}°C")

if __name__ == "__main__":
    main()
