#!/usr/bin/env python3
"""
Training script for StableProt V8 Disjoint Multi-Head Architecture.
Implements decoupled checkpoints (model_tm.pt, model_ogt.pt), independent schedulers,
auxiliary projection bottleneck (Linear(9,64) Tm / Linear(8,64) OGT), bounded NLL variance,
true val_ogt split loading, and scheduled OGT noise injection.
"""

import os
import sys
import argparse
from pathlib import Path
from itertools import cycle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "src" / "training" / "v10"))
from config import CONFIG

def compute_aa_composition(sequences):
    comp_list = []
    for seq in sequences:
        s_clean = "".join([c for c in str(seq).upper() if c.isupper() and c.isalpha()])
        L = max(len(s_clean), 1)
        f_charged = (s_clean.count("D") + s_clean.count("E") + s_clean.count("K") + s_clean.count("R")) / L
        f_hydro = (s_clean.count("A") + s_clean.count("I") + s_clean.count("L") + s_clean.count("M") + s_clean.count("F") + s_clean.count("W") + s_clean.count("V")) / L
        f_proline = s_clean.count("P") / L
        f_cysteine = s_clean.count("C") / L
        f_aromatic = (s_clean.count("F") + s_clean.count("W") + s_clean.count("Y")) / L
        v = s_clean.count("V")
        il = s_clean.count("I") + s_clean.count("L")
        aliph_raw = (s_clean.count("A") + 2.9 * v + 3.9 * il) / (L * 100.0)
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

def enrich_inputs(embeddings, sequences, tmhmm_flags=None, ogt_priors=None, ogt_sigmas=None):
    """Return tuple (embeddings_1280, aux_features_9_or_8)."""
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
        if ogt_sigmas is not None:
            # The prior's own uncertainty, so the Tm pathway can weigh how far to trust it.
            # sigma_OGT ranks prior quality on Tm proteins (Spearman 0.41 against the realised
            # OGT error; quartile means 12.0, 12.5, 17.0, 24.0 °C), which a constant prior cannot.
            sig_t = torch.tensor([float(s) / 100.0 for s in ogt_sigmas], dtype=torch.float32).unsqueeze(-1)
            aux = torch.cat([ogt_t, sig_t, tm_t, lengths_t, aa_comp], dim=-1)
        else:
            aux = torch.cat([ogt_t, tm_t, lengths_t, aa_comp], dim=-1)
    else:
        aux = torch.cat([tm_t, lengths_t, aa_comp], dim=-1)
    return embeddings.float(), aux

class MultiHeadSaProtV8(nn.Module):
    def __init__(self, emb_dim=1280, aux_dim_tm=9, aux_dim_ogt=8, proj_dim=64, hidden1=512, hidden2=256, dropout1=0.3, dropout2=0.2, use_residuals=True, ogt_heteroscedastic=False):
        super().__init__()
        self.proj_dim = proj_dim
        self.use_residuals = use_residuals
        # Default False keeps head_ogt at width 1 so pre-existing checkpoints load unchanged.
        self.ogt_heteroscedastic = ogt_heteroscedastic
        if proj_dim is not None:
            self.aux_proj_tm = nn.Sequential(nn.Linear(aux_dim_tm, proj_dim), nn.GELU(), nn.LayerNorm(proj_dim))
            self.aux_proj_ogt = nn.Sequential(nn.Linear(aux_dim_ogt, proj_dim), nn.GELU(), nn.LayerNorm(proj_dim))
            in_dim_tm = emb_dim + proj_dim
            in_dim_ogt = emb_dim + proj_dim
        else:
            self.aux_proj_tm = None
            self.aux_proj_ogt = None
            in_dim_tm = emb_dim + aux_dim_tm
            in_dim_ogt = emb_dim + aux_dim_ogt
            
        # Disjoint Tm pathway
        self.fc1_tm = nn.Linear(in_dim_tm, hidden1)
        self.ln1_tm = nn.LayerNorm(hidden1)
        self.fc2_tm = nn.Linear(hidden1, hidden2)
        self.ln2_tm = nn.LayerNorm(hidden2)
        if use_residuals:
            self.res_tm = nn.Linear(hidden1, hidden2)
        else:
            self.register_parameter('res_tm', None)
        self.head_tm = nn.Linear(hidden2, 2)
        
        # Disjoint OGT pathway
        self.fc1_ogt = nn.Linear(in_dim_ogt, hidden1)
        self.ln1_ogt = nn.LayerNorm(hidden1)
        self.fc2_ogt = nn.Linear(hidden1, hidden2)
        self.ln2_ogt = nn.LayerNorm(hidden2)
        if use_residuals:
            self.res_ogt = nn.Linear(hidden1, hidden2)
        else:
            self.register_parameter('res_ogt', None)
        self.head_ogt = nn.Linear(hidden2, 2 if ogt_heteroscedastic else 1)
        
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout1)
        self.drop2 = nn.Dropout(dropout2)
        
    def forward(self, x_emb, x_aux=None, head='tm'):
        if x_aux is None:
            if head == 'tm':
                x_aux = x_emb[:, -self.aux_proj_tm[0].in_features:] if self.aux_proj_tm is not None else x_emb[:, 1280:]
                x_emb = x_emb[:, :1280]
            else:
                x_aux = x_emb[:, -8:]
                x_emb = x_emb[:, :1280]
                
        if head == 'tm':
            h_aux = self.aux_proj_tm(x_aux) if self.aux_proj_tm is not None else x_aux
            h = torch.cat([x_emb, h_aux], dim=-1)
            h1 = self.drop1(self.act(self.ln1_tm(self.fc1_tm(h))))
            if self.res_tm is not None:
                h2 = self.drop2(self.act(self.ln2_tm(self.fc2_tm(h1)) + self.res_tm(h1)))
            else:
                h2 = self.drop2(self.act(self.ln2_tm(self.fc2_tm(h1))))
            out = self.head_tm(h2)
            z_mean = out[:, 0]
            z_var = F.softplus(out[:, 1]) + 1e-4
            return z_mean, z_var
        else:
            h_aux = self.aux_proj_ogt(x_aux) if self.aux_proj_ogt is not None else x_aux
            h = torch.cat([x_emb, h_aux], dim=-1)
            h1 = self.drop1(self.act(self.ln1_ogt(self.fc1_ogt(h))))
            if self.res_ogt is not None:
                h2 = self.drop2(self.act(self.ln2_ogt(self.fc2_ogt(h1)) + self.res_ogt(h1)))
            else:
                h2 = self.drop2(self.act(self.ln2_ogt(self.fc2_ogt(h1))))
            out = self.head_ogt(h2)
            if self.ogt_heteroscedastic:
                return out[:, 0], F.softplus(out[:, 1]) + 1e-4
            return out.squeeze(-1)

class ProteinDataset(Dataset):
    def __init__(self, emb, aux, y, weights=None, augment=False, augment_prob=0.15, augment_noise_std=0.02):
        self.emb = emb
        self.aux = aux
        self.y = y
        self.weights = weights if weights is not None else torch.ones_like(y)
        self.augment = augment
        self.augment_prob = augment_prob
        self.augment_noise_std = augment_noise_std
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        e, a, y, w = self.emb[idx], self.aux[idx], self.y[idx], self.weights[idx]
        if self.augment and torch.rand(1).item() < self.augment_prob:
            noise = torch.randn_like(e) * self.augment_noise_std
            e = e + noise
        return e, a, y, w

def compute_bin_weights(labels, bin_edges, clamp_min=0.3, clamp_max=22.0, power=0.75):
    labels_np = labels.numpy() if hasattr(labels, 'numpy') else np.array(labels)
    bin_idx = np.digitize(labels_np, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, len(bin_edges) - 2)
    counts = np.bincount(bin_idx, minlength=len(bin_edges)-1).astype(float)
    counts[counts == 0] = 1.0
    med = np.median(counts[counts > 0])
    w = np.clip((med / counts) ** power, clamp_min, clamp_max)
    return torch.tensor(w[bin_idx], dtype=torch.float32)

def build_tm_iqr_weights(train_sequences):
    """Compute inverse-IQR sample weights from raw cross-database Tm readings."""
    import pandas as pd
    
    meltome = pd.read_csv(str(PROJECT_ROOT / 'data/flip_meltome/mixed_split.csv'))
    tember_files = [
        PROJECT_ROOT / 'benchmark_models_tm/TemBERTure_repo/data/TemBERTureTrain_reg.txt',
        PROJECT_ROOT / 'benchmark_models_tm/TemBERTure_repo/data/TemBERTureVal_reg.txt',
        PROJECT_ROOT / 'benchmark_models_tm/TemBERTure_repo/data/TemBERTureTest_reg.txt',
    ]
    tember = pd.concat([pd.read_csv(str(f)) for f in tember_files], ignore_index=True)
    
    m_sub = meltome[['sequence', 'label']].rename(columns={'label': 'tm'})
    t_sub = tember[['Sequence', 'Tm']].rename(columns={'Sequence': 'sequence', 'Tm': 'tm'})
    combined = pd.concat([m_sub, t_sub], ignore_index=True)
    
    def iqr(x):
        return np.percentile(x, 75) - np.percentile(x, 25)
    
    seq_iqr = combined.groupby('sequence')['tm'].agg(['count', iqr])
    seq_iqr.columns = ['count', 'iqr']
    seq_iqr_map = seq_iqr[seq_iqr['count'] > 1]['iqr'].to_dict()
    
    weights = []
    matched = 0
    iqr_scale = CONFIG.get('iqr_scale', 6.34)
    iqr_impute = CONFIG.get('iqr_impute_val', 0.62)
    for seq in train_sequences:
        s = str(seq)
        if s in seq_iqr_map:
            w = 1.0 / (1.0 + (seq_iqr_map[s] / iqr_scale) ** 2)
            matched += 1
        else:
            w = 1.0 / (1.0 + (iqr_impute / iqr_scale) ** 2)
        weights.append(w)
    
    print(f"  IQR weights: {matched}/{len(train_sequences)} sequences matched ({100*matched/len(train_sequences):.1f}%)")
    return torch.tensor(weights, dtype=torch.float32)

def focal_huber_loss(pred, target, delta=15.0, gamma=2.0, beta=0.5, weights=None):
    """Focal regression: upweight hard examples (large errors) with Huber base."""
    huber = F.huber_loss(pred, target, delta=delta, reduction='none')
    error = torch.abs(pred - target).detach()
    focal_weight = (error / (error + beta)) ** gamma
    loss = huber * focal_weight
    if weights is not None:
        loss = loss * weights
    return loss.mean()

class MesophilicSubsampler:
    """Per-epoch random subsampler that keeps mesophilic OGT samples."""
    def __init__(self, labels, meso_low=25.0, meso_high=40.0, keep_rate=0.14):
        labels_np = labels.numpy() if hasattr(labels, 'numpy') else np.array(labels)
        self.meso_mask = (labels_np >= meso_low) & (labels_np <= meso_high)
        self.non_meso_idx = np.where(~self.meso_mask)[0]
        self.meso_idx = np.where(self.meso_mask)[0]
        self.keep_rate = keep_rate
        self.n_meso_keep = int(len(self.meso_idx) * keep_rate)
        print(f"  MesophilicSubsampler: {len(self.meso_idx)} mesophilic ({meso_low}-{meso_high}°C), "
              f"keeping {self.n_meso_keep}/epoch + {len(self.non_meso_idx)} non-mesophilic")
    
    def sample_epoch_indices(self):
        """Return shuffled indices for one epoch."""
        meso_sample = np.random.choice(self.meso_idx, size=self.n_meso_keep, replace=False)
        all_idx = np.concatenate([self.non_meso_idx, meso_sample])
        np.random.shuffle(all_idx)
        return all_idx

def sanitize_data(dict_data, is_tm=True):
    seqs = dict_data['sequences']
    embs = dict_data['embeddings']
    if is_tm:
        lbls = dict_data.get('tm_consensus', dict_data.get('labels'))
        ogts = dict_data.get('ogt', [50.0]*len(seqs))
    else:
        lbls = dict_data.get('ogt_consensus', dict_data.get('labels'))
        ogts = lbls
        
    keep_indices = []
    min_len = CONFIG.get('seq_len_min', 50)
    for idx, seq in enumerate(seqs):
        s_clean = "".join([c for c in str(seq).upper() if c.isupper() and c.isalpha()])
        if len(s_clean) < min_len:
            continue
        if is_tm and ogts is not None and idx < len(ogts):
            o_val = float(ogts[idx]) if ogts[idx] is not None else 0.0
            if float(lbls[idx]) < o_val:
                continue
        keep_indices.append(idx)
        
    print(f"  Sanitization: kept {len(keep_indices)} / {len(seqs)} sequences.")
    embs_sub = embs[keep_indices]
    seqs_sub = [seqs[i] for i in keep_indices]
    lbls_sub = torch.tensor([float(lbls[i]) for i in keep_indices], dtype=torch.float32)
    tmhmm_sub = [dict_data.get('tmhmm_tm_binary', [0]*len(seqs))[i] for i in keep_indices]
    
    if is_tm:
        ogts_sub = [ogts[i] for i in keep_indices]
        return embs_sub, seqs_sub, lbls_sub, ogts_sub, tmhmm_sub
    else:
        return embs_sub, seqs_sub, lbls_sub, tmhmm_sub

def train_seed(seed, train_tm_ds, train_ogt_ds, val_tm_ds, val_ogt_ds, tm_mean, tm_std, ogt_mean, ogt_std, ogt_subsampler, device, save_dir, max_epochs, patience, train_tm=True, train_ogt=True):
    print(f"\n{'='*50}\nTraining V9 Disjoint Architecture (Seed {seed})\n{'='*50}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    ogt_hetero = CONFIG.get('ogt_heteroscedastic', False)
    ogt_loss_mode = CONFIG.get('ogt_loss_mode', 'huber_nll')
    model = MultiHeadSaProtV8(
        aux_dim_tm=10 if CONFIG.get('tm_use_ogt_sigma', False) else 9,
        proj_dim=CONFIG.get('proj_dim', 64),
        use_residuals=CONFIG.get('use_residuals', True),
        ogt_heteroscedastic=ogt_hetero
    ).to(device)

    def ogt_loss_fn(pred, target, weights, std):
        """OGT objective. 'huber_nll' keeps the point loss that produced the v9 accuracy and
        trains the variance against a detached mean, so adding uncertainty cannot move the
        point predictions. 'nll' replaces the point loss entirely."""
        if not ogt_hetero:
            return focal_huber_loss(pred, target, delta=CONFIG['huber_delta_ogt'] / std,
                                    gamma=CONFIG['focal_gamma'], beta=CONFIG['focal_beta'],
                                    weights=weights)
        z_mu, z_var = pred
        if ogt_loss_mode == 'nll':
            return (( 0.5 * (z_mu - target)**2 / z_var + 0.5 * torch.log(z_var)) * weights).mean()
        point = focal_huber_loss(z_mu, target, delta=CONFIG['huber_delta_ogt'] / std,
                                 gamma=CONFIG['focal_gamma'], beta=CONFIG['focal_beta'],
                                 weights=weights)
        resid = (z_mu.detach() - target)**2
        var_nll = ((0.5 * resid / z_var + 0.5 * torch.log(z_var)) * weights).mean()
        return point + CONFIG.get('ogt_var_loss_weight', 1.0) * var_nll

    tm_loss_mode = CONFIG.get('tm_loss_mode', 'nll')

    def tm_loss_fn(z_mu, z_var, target, weights):
        """Tm objective. Under 'nll' the joint Gaussian likelihood lets the variance be driven by
        the same gradient as the mean, and it degenerates into a monotone function of the mean.
        'huber_nll' detaches the mean inside the variance term so the variance is fitted to the
        realised squared residual instead."""
        if tm_loss_mode == 'nll':
            return ((0.5 * (z_mu - target)**2 / z_var + 0.5 * torch.log(z_var)) * weights).mean()
        point = focal_huber_loss(z_mu, target, delta=CONFIG['huber_delta_tm'] / tm_std,
                                 gamma=CONFIG['focal_gamma'], beta=CONFIG['focal_beta'],
                                 weights=weights)
        resid = (z_mu.detach() - target)**2
        var_nll = ((0.5 * resid / z_var + 0.5 * torch.log(z_var)) * weights).mean()
        return point + CONFIG.get('tm_var_loss_weight', 1.0) * var_nll

    
    params_tm = [
        {'params': model.aux_proj_tm.parameters(), 'lr': CONFIG['learning_rate']} if model.aux_proj_tm is not None else None,
        {'params': model.fc1_tm.parameters(), 'lr': CONFIG['learning_rate']},
        {'params': model.ln1_tm.parameters(), 'lr': CONFIG['learning_rate']},
        {'params': model.fc2_tm.parameters(), 'lr': CONFIG['learning_rate']},
        {'params': model.ln2_tm.parameters(), 'lr': CONFIG['learning_rate']},
        {'params': model.res_tm.parameters(), 'lr': CONFIG['learning_rate']} if model.res_tm is not None else None,
        {'params': model.head_tm.parameters(), 'lr': CONFIG['learning_rate']},
    ]
    opt_tm = optim.AdamW([p for p in params_tm if p is not None], weight_decay=CONFIG['weight_decay'])
    
    params_ogt = [
        {'params': model.aux_proj_ogt.parameters(), 'lr': CONFIG['learning_rate']} if model.aux_proj_ogt is not None else None,
        {'params': model.fc1_ogt.parameters(), 'lr': CONFIG['learning_rate']},
        {'params': model.ln1_ogt.parameters(), 'lr': CONFIG['learning_rate']},
        {'params': model.fc2_ogt.parameters(), 'lr': CONFIG['learning_rate']},
        {'params': model.ln2_ogt.parameters(), 'lr': CONFIG['learning_rate']},
        {'params': model.res_ogt.parameters(), 'lr': CONFIG['learning_rate']} if model.res_ogt is not None else None,
        {'params': model.head_ogt.parameters(), 'lr': CONFIG['learning_rate']},
    ]
    opt_ogt = optim.AdamW([p for p in params_ogt if p is not None], weight_decay=CONFIG['weight_decay'])
    
    sched_tm = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt_tm, T_0=CONFIG.get('scheduler_T0', 10), T_mult=CONFIG.get('scheduler_Tmult', 2), eta_min=1e-6)
    sched_ogt = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt_ogt, T_0=CONFIG.get('scheduler_T0', 10), T_mult=CONFIG.get('scheduler_Tmult', 2), eta_min=1e-6)
    scaler_ogt = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    scaler_tm = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    tm_loader = DataLoader(train_tm_ds, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    ogt_loader = DataLoader(train_ogt_ds, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    val_tm_loader = DataLoader(val_tm_ds, batch_size=CONFIG['batch_size'], shuffle=False)
    val_ogt_loader = DataLoader(val_ogt_ds, batch_size=CONFIG['batch_size'], shuffle=False)
    
    tm_iter = iter(cycle(tm_loader))
    best_tm_mae = float('inf')
    best_ogt_mae = float('inf')
    patience_tm = 0
    patience_ogt = 0
    path_tm = os.path.join(save_dir, 'model_tm.pt')
    path_ogt = os.path.join(save_dir, 'model_ogt.pt')
    
    for epoch in range(max_epochs):
        model.train()
        train_tm_loss = 0.0
        train_ogt_loss = 0.0
        
        # Per-epoch OGT subsampling
        if ogt_subsampler is not None:
            epoch_idx = ogt_subsampler.sample_epoch_indices()
            epoch_ogt_ds = torch.utils.data.Subset(train_ogt_ds, epoch_idx)
            ogt_loader = DataLoader(epoch_ogt_ds, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
        
        tm_iter = iter(cycle(tm_loader))
        
        pbar = tqdm(ogt_loader, desc=f"Seed {seed} | Ep {epoch+1}/{max_epochs}", leave=False)
        for emb_ogt, aux_ogt, y_ogt, w_ogt in pbar:
            # 1. OGT Step. Skipped when the OGT head is being used as a frozen prior source, so
            # the prior the Tm head trained against stays the prior it will receive at inference.
            if not train_ogt:
                emb_tm, aux_tm, y_tm, w_tm = next(tm_iter)
                emb_tm, aux_tm, y_tm, w_tm = emb_tm.to(device), aux_tm.to(device), y_tm.to(device), w_tm.to(device)
                if CONFIG.get('target_jitter_std', 0.5) > 0:
                    y_tm = y_tm + torch.randn_like(y_tm) * CONFIG.get('target_jitter_std', 0.5)
                if CONFIG.get('tm_ogt_noise_std', 0.0) > 0:
                    aux_tm = aux_tm.clone()
                    aux_tm[:, 0] = aux_tm[:, 0] + (torch.randn_like(aux_tm[:, 0]) * CONFIG['tm_ogt_noise_std']) / 100.0
                z_tm = (y_tm - tm_mean) / tm_std
                opt_tm.zero_grad()
                if scaler_tm:
                    with torch.amp.autocast('cuda'):
                        z_mu, z_var = model(emb_tm, aux_tm, head='tm')
                        loss_tm = tm_loss_fn(z_mu, z_var, z_tm, w_tm)
                    scaler_tm.scale(loss_tm).backward()
                    scaler_tm.unscale_(opt_tm)
                    nn.utils.clip_grad_norm_([p for g in opt_tm.param_groups for p in g['params']],
                                             CONFIG['grad_clip_max_norm'])
                    scaler_tm.step(opt_tm); scaler_tm.update()
                else:
                    z_mu, z_var = model(emb_tm, aux_tm, head='tm')
                    loss_tm = tm_loss_fn(z_mu, z_var, z_tm, w_tm)
                    loss_tm.backward()
                    nn.utils.clip_grad_norm_([p for g in opt_tm.param_groups for p in g['params']],
                                             CONFIG['grad_clip_max_norm'])
                    opt_tm.step()
                train_tm_loss += loss_tm.item()
                continue
            emb_ogt, aux_ogt, y_ogt, w_ogt = emb_ogt.to(device), aux_ogt.to(device), y_ogt.to(device), w_ogt.to(device)
            if CONFIG.get('target_jitter_std', 0.5) > 0:
                y_ogt_raw = y_ogt * ogt_std + ogt_mean
                y_ogt_raw = y_ogt_raw + torch.randn_like(y_ogt_raw) * CONFIG['target_jitter_std']
                y_ogt = (y_ogt_raw - ogt_mean) / ogt_std
                
            opt_ogt.zero_grad()
            if scaler_ogt:
                with torch.amp.autocast('cuda'):
                    loss_ogt = ogt_loss_fn(model(emb_ogt, aux_ogt, head='ogt'), y_ogt, w_ogt, ogt_std)
                scaler_ogt.scale(loss_ogt).backward()
                scaler_ogt.unscale_(opt_ogt)
                ogt_params = [p for g in opt_ogt.param_groups for p in g['params']]
                nn.utils.clip_grad_norm_(ogt_params, CONFIG['grad_clip_max_norm'])
                scaler_ogt.step(opt_ogt)
                scaler_ogt.update()
            else:
                loss_ogt = ogt_loss_fn(model(emb_ogt, aux_ogt, head='ogt'), y_ogt, w_ogt, ogt_std)
                loss_ogt.backward()
                ogt_params = [p for g in opt_ogt.param_groups for p in g['params']]
                nn.utils.clip_grad_norm_(ogt_params, CONFIG['grad_clip_max_norm'])
                opt_ogt.step()
            train_ogt_loss += loss_ogt.item()

            # 2. Tm Step
            if not train_tm:
                continue
            emb_tm, aux_tm, y_tm, w_tm = next(tm_iter)
            emb_tm, aux_tm, y_tm, w_tm = emb_tm.to(device), aux_tm.to(device), y_tm.to(device), w_tm.to(device)
            if CONFIG.get('target_jitter_std', 0.5) > 0:
                y_tm = y_tm + torch.randn_like(y_tm) * CONFIG.get('target_jitter_std', 0.5)
            if CONFIG.get('tm_ogt_noise_std', 6.0) > 0:
                aux_tm = aux_tm.clone()
                aux_tm[:, 0] = aux_tm[:, 0] + (torch.randn_like(aux_tm[:, 0]) * CONFIG.get('tm_ogt_noise_std', 6.0)) / 100.0
                
            if CONFIG.get('mixup_alpha', 0) > 0:
                lam = np.random.beta(CONFIG['mixup_alpha'], CONFIG['mixup_alpha'])
                idx = torch.randperm(emb_tm.size(0))
                emb_tm = lam * emb_tm + (1 - lam) * emb_tm[idx]
                y_tm = lam * y_tm + (1 - lam) * y_tm[idx]
                w_tm = lam * w_tm + (1 - lam) * w_tm[idx]
                
            z_tm = (y_tm - tm_mean) / tm_std
            
            opt_tm.zero_grad()
            if scaler_tm:
                with torch.amp.autocast('cuda'):
                    z_mu, z_var = model(emb_tm, aux_tm, head='tm')
                    loss_tm = tm_loss_fn(z_mu, z_var, z_tm, w_tm)
                scaler_tm.scale(loss_tm).backward()
                scaler_tm.unscale_(opt_tm)
                tm_params = [p for g in opt_tm.param_groups for p in g['params']]
                nn.utils.clip_grad_norm_(tm_params, CONFIG['grad_clip_max_norm'])
                scaler_tm.step(opt_tm)
                scaler_tm.update()
            else:
                z_mu, z_var = model(emb_tm, aux_tm, head='tm')
                loss_tm = tm_loss_fn(z_mu, z_var, z_tm, w_tm)
                loss_tm.backward()
                tm_params = [p for g in opt_tm.param_groups for p in g['params']]
                nn.utils.clip_grad_norm_(tm_params, CONFIG['grad_clip_max_norm'])
                opt_tm.step()
            train_tm_loss += loss_tm.item()
            
        # Validation
        model.eval()
        val_tm_mae = float('nan')
        if train_tm:
            val_tm_mae = 0.0
            with torch.no_grad():
                for emb, aux, y, _ in val_tm_loader:
                    emb, aux, y = emb.to(device), aux.to(device), y.to(device)
                    z_mu, _ = model(emb, aux, head='tm')
                    pred_y = z_mu * tm_std + tm_mean
                    val_tm_mae += torch.abs(pred_y - y).sum().item()
            val_tm_mae /= len(val_tm_loader.dataset)
        
        val_ogt_mae = float('nan')
        if train_ogt:
            val_ogt_mae = 0.0
            with torch.no_grad():
                for emb, aux, y, _ in val_ogt_loader:
                    emb, aux, y = emb.to(device), aux.to(device), y.to(device)
                    pred_o = model(emb, aux, head='ogt')
                    if ogt_hetero:
                        pred_o = pred_o[0]
                    pred_o_raw = pred_o * ogt_std + ogt_mean
                    val_ogt_mae += torch.abs(pred_o_raw - y).sum().item()
            val_ogt_mae /= len(val_ogt_loader.dataset)

        if train_tm:
            sched_tm.step()
        if train_ogt:
            sched_ogt.step()

        print(f"  Ep {epoch+1:3d} | Val Tm MAE: {val_tm_mae:.4f}°C | Val OGT MAE: {val_ogt_mae:.4f}°C")

        if not train_tm:
            patience_tm = patience
        elif val_tm_mae < best_tm_mae:
            best_tm_mae = val_tm_mae
            patience_tm = 0
            torch.save(model.state_dict(), path_tm)
        else:
            patience_tm += 1


        if not train_ogt:
            patience_ogt = patience
        elif val_ogt_mae < best_ogt_mae:
            best_ogt_mae = val_ogt_mae
            patience_ogt = 0
            torch.save(model.state_dict(), path_ogt)
        else:
            patience_ogt += 1
            
        if patience_tm >= patience and patience_ogt >= patience:
            print(f"  Early stopping triggered for both heads after {epoch+1} epochs.")
            break
            
    return path_tm, best_tm_mae, path_ogt, best_ogt_mae

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Run quick debug test on subset")
    parser.add_argument("--ogt-only", action="store_true",
                        help="Train only the OGT pathway. The pathways share no parameters, so "
                             "existing Tm checkpoints are left untouched.")
    parser.add_argument("--tm-only", action="store_true",
                        help="Train only the Tm pathway, leaving the OGT head frozen as the "
                             "prior source. Required when tm_prior_source = 'predicted'.")
    parser.add_argument("--save-dir", default=None, help="Override the results directory")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Debug mode: {args.debug}")
    
    data_path = "data/embeddings/saprot_tm_struct_embeddings.pt"
    print(f"Loading master dataset from {data_path}...")
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    
    print("Loading true OGT validation set...")
    ogt_split = torch.load("data/embeddings/prepared_data_v7_saprot1.3b_seqonly_ogt_split.pt", map_location='cpu', weights_only=False)
    data['val_ogt'] = ogt_split['val_ogt']
    
    if args.debug:
        print("Debug mode enabled: slicing dictionaries to 1% subset before sanitization...")
        for k in ['embeddings', 'sequences', 'tm_consensus', 'ogt', 'tmhmm_tm_binary', 'ids']:
            if k in data['train_tm']: data['train_tm'][k] = data['train_tm'][k][:200]
            if k in data['val_tm']: data['val_tm'][k] = data['val_tm'][k][:50]
        for k in ['embeddings', 'sequences', 'ogt_consensus', 'ogt_original', 'tmhmm_tm_binary', 'ids']:
            if k in data['train_ogt']: data['train_ogt'][k] = data['train_ogt'][k][:1000]
            if k in data['val_ogt']: data['val_ogt'][k] = data['val_ogt'][k][:200]

    print("Sanitizing datasets...")
    tr_tm_emb, tr_tm_seq, tr_tm_lbl, tr_tm_ogt, tr_tm_tmhmm = sanitize_data(data['train_tm'], is_tm=True)
    val_tm_emb, val_tm_seq, val_tm_lbl, val_tm_ogt, val_tm_tmhmm = sanitize_data(data['val_tm'], is_tm=True)
    tr_ogt_emb, tr_ogt_seq, tr_ogt_lbl, tr_ogt_tmhmm = sanitize_data(data['train_ogt'], is_tm=False)
    val_ogt_emb, val_ogt_seq, val_ogt_lbl, val_ogt_tmhmm = sanitize_data(data['val_ogt'], is_tm=False)
    
    if args.debug:
        seeds = [1]
        max_epochs = 2
        patience = 2
    else:
        seeds = CONFIG['seeds']
        max_epochs = CONFIG['max_epochs']
        patience = CONFIG['early_stopping_patience']
        
    tm_mean = tr_tm_lbl.mean().item()
    tm_std = tr_tm_lbl.std().item()
    ogt_mean = tr_ogt_lbl.mean().item()
    ogt_std = tr_ogt_lbl.std().item()
    print(f"Target Tm Z-score scaling  | Mean: {tm_mean:.2f}°C, Std: {tm_std:.2f}°C")
    print(f"Target OGT Z-score scaling | Mean: {ogt_mean:.2f}°C, Std: {ogt_std:.2f}°C")
    
    tr_tm_sig = val_tm_sig = None
    if CONFIG.get('tm_prior_source', 'true') == 'predicted':
        # Train the Tm pathway on the prior it will actually receive at inference. Passing the
        # true OGT during training and a prediction at test time is a train/serve mismatch worth
        # 1.3 C: the prediction is off by ~17 C on Tm proteins (r = 0.34 against truth).
        ckpt_dir = os.path.join(PROJECT_ROOT, CONFIG.get('tm_prior_ckpt_dir',
                                'experiments/src/training/v10/results'))
        print(f"Predicting the OGT prior with the frozen ensemble at {ckpt_dir} ...")
        pn = torch.load(os.path.join(ckpt_dir, 'normalization_stats.pt'), map_location='cpu',
                        weights_only=False)
        def predict_prior(embs, seqs, flags):
            e_, a_ = enrich_inputs(embs, seqs, flags)
            MU, VA = [], []
            for sd in CONFIG['seeds']:
                p = os.path.join(ckpt_dir, f"seed{sd}", "model_ogt.pt")
                if not os.path.exists(p):
                    continue
                net = MultiHeadSaProtV8(ogt_heteroscedastic=True).to(device)
                net.load_state_dict(torch.load(p, map_location=device, weights_only=False))
                net.eval()
                mus, vas = [], []
                with torch.no_grad():
                    for i in range(0, len(e_), 4096):
                        zm, zv = net(e_[i:i+4096].to(device), a_[i:i+4096].to(device), head='ogt')
                        mus.append(zm.cpu()); vas.append(zv.cpu())
                MU.append((torch.cat(mus) * pn['ogt_std'] + pn['ogt_mean']).numpy())
                VA.append((torch.cat(vas) * pn['ogt_std'] ** 2).numpy())
            MU = np.stack(MU); VA = np.stack(VA)
            return MU.mean(0), np.sqrt(VA.mean(0) + MU.var(0))
        tr_tm_ogt, tr_tm_sig = predict_prior(tr_tm_emb, tr_tm_seq, tr_tm_tmhmm)
        val_tm_ogt, val_tm_sig = predict_prior(val_tm_emb, val_tm_seq, val_tm_tmhmm)
        print(f"  train prior: mean {tr_tm_ogt.mean():.1f} C, mean sigma {tr_tm_sig.mean():.1f} C")
        print(f"  val   prior: mean {val_tm_ogt.mean():.1f} C, mean sigma {val_tm_sig.mean():.1f} C")
        if not CONFIG.get('tm_use_ogt_sigma', False):
            tr_tm_sig = val_tm_sig = None

    print("Enriching features (1280-dim embedding + aux features)...")
    tr_tm_e, tr_tm_aux = enrich_inputs(tr_tm_emb, tr_tm_seq, tr_tm_tmhmm, tr_tm_ogt, tr_tm_sig)
    val_tm_e, val_tm_aux = enrich_inputs(val_tm_emb, val_tm_seq, val_tm_tmhmm, val_tm_ogt, val_tm_sig)
    tr_ogt_e, tr_ogt_aux = enrich_inputs(tr_ogt_emb, tr_ogt_seq, tr_ogt_tmhmm)
    val_ogt_e, val_ogt_aux = enrich_inputs(val_ogt_emb, val_ogt_seq, val_ogt_tmhmm)
    
    print("Computing inverse-IQR weights from raw Meltome + TemBERTure data...")
    tm_iqr_weights = build_tm_iqr_weights(tr_tm_seq)
    tr_tm_w = compute_bin_weights(tr_tm_lbl, CONFIG['bin_edges'],
        CONFIG['weight_clamp_min'], CONFIG['weight_clamp_max'], CONFIG.get('weight_power', 0.75))
    tr_tm_w = tr_tm_w * tm_iqr_weights
    val_tm_w = torch.ones_like(val_tm_lbl)
    
    tr_ogt_w = compute_bin_weights(tr_ogt_lbl, CONFIG['bin_edges'],
        CONFIG['weight_clamp_min'], CONFIG['weight_clamp_max'], CONFIG.get('weight_power', 0.75))
    val_ogt_w = torch.ones_like(val_ogt_lbl)
    
    if CONFIG.get('ogt_normalize', False):
        tr_ogt_lbl_z = (tr_ogt_lbl - ogt_mean) / ogt_std
    else:
        tr_ogt_lbl_z = tr_ogt_lbl
        
    train_tm_ds = ProteinDataset(tr_tm_e, tr_tm_aux, tr_tm_lbl, tr_tm_w, augment=True)
    val_tm_ds = ProteinDataset(val_tm_e, val_tm_aux, val_tm_lbl, val_tm_w)
    train_ogt_ds = ProteinDataset(tr_ogt_e, tr_ogt_aux, tr_ogt_lbl_z, tr_ogt_w, augment=True)
    val_ogt_ds = ProteinDataset(val_ogt_e, val_ogt_aux, val_ogt_lbl, val_ogt_w)
    
    ogt_subsampler = MesophilicSubsampler(
        tr_ogt_lbl,
        keep_rate=CONFIG.get('ogt_subsample_meso_rate', 0.14)
    ) if CONFIG.get('ogt_subsample_meso_rate', 1.0) < 1.0 else None
    
    save_dir = args.save_dir or os.path.join(PROJECT_ROOT, "experiments/src/training/v10/results")
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save({'tm_mean': tm_mean, 'tm_std': tm_std, 'ogt_mean': ogt_mean, 'ogt_std': ogt_std},
               os.path.join(save_dir, 'normalization_stats.pt'))
    
    results = {}
    for seed in seeds:
        seed_dir = os.path.join(save_dir, f"seed{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        path_tm, best_tm, path_ogt, best_ogt = train_seed(
            seed, train_tm_ds, train_ogt_ds, val_tm_ds, val_ogt_ds,
            tm_mean, tm_std, ogt_mean, ogt_std, ogt_subsampler, device, seed_dir, max_epochs, patience,
            train_tm=not args.ogt_only, train_ogt=not args.tm_only
        )
        results[seed] = {'best_val_tm_mae': best_tm, 'best_val_ogt_mae': best_ogt}
        print(f"Seed {seed} complete | Best Val Tm MAE: {best_tm:.4f}°C | Best Val OGT MAE: {best_ogt:.4f}°C")
        
    avg_tm = np.mean([r['best_val_tm_mae'] for r in results.values()])
    avg_ogt = np.mean([r['best_val_ogt_mae'] for r in results.values()])
    print(f"\n{'='*50}\nTraining Complete. 5-Seed Ensemble Summary:\nAvg Val Tm MAE: {avg_tm:.4f}°C | Avg Val OGT MAE: {avg_ogt:.4f}°C\n{'='*50}")

if __name__ == "__main__":
    main()
