#!/usr/bin/env python3
"""
Isolated Model Calibration Sweep Runner (`experiments/src/model_calibration/sweep_runner.py`)

Self-contained hyperparameter execution engine with full support across all 32 parameters in 8 groups.
Features in-memory caching of enriched inputs and string compositions so that 150+ runs bypass
string processing loops and execute GPU epochs instantly (`~6s per run`).
"""

import os
import sys
import csv
import time
import random
import argparse
import copy
from pathlib import Path
from itertools import cycle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
VERSION = os.environ.get("STABLEPROT_VERSION", "v8_disjoint")
sys.path.append(str(PROJECT_ROOT / "experiments" / "src" / "training" / VERSION))
from config import CONFIG as BASE_CONFIG
from train import (
    compute_aa_composition, enrich_inputs, build_tm_iqr_weights, focal_huber_loss, MesophilicSubsampler, ProteinDataset
)

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_bin_weights_calib(labels, bin_edges, clamp_min=0.3, clamp_max=22.0, power=0.75):
    """Compute bin weighting, returning all ones if power == 0.0 (no weighting baseline)."""
    labels_np = labels.numpy() if hasattr(labels, 'numpy') else np.array(labels)
    if power <= 0.001:
        return torch.ones(len(labels_np), dtype=torch.float32)
    bin_idx = np.digitize(labels_np, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, len(bin_edges) - 2)
    counts = np.bincount(bin_idx, minlength=len(bin_edges)-1).astype(float)
    counts[counts == 0] = 1.0
    med = np.median(counts[counts > 0])
    w = np.clip((med / counts) ** power, clamp_min, clamp_max)
    return torch.tensor(w[bin_idx], dtype=torch.float32)

_seq_iqr_map_cached = None

def get_seq_iqr_map():
    global _seq_iqr_map_cached
    if _seq_iqr_map_cached is not None:
        return _seq_iqr_map_cached
    import pandas as pd
    def iqr(x):
        return np.percentile(x, 75) - np.percentile(x, 25)
    
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
    seq_iqr = combined.groupby('sequence')['tm'].agg(['count', iqr])
    seq_iqr.columns = ['count', 'iqr']
    _seq_iqr_map_cached = seq_iqr[seq_iqr['count'] > 1]['iqr'].to_dict()
    return _seq_iqr_map_cached

def get_clean_sequences(dict_data):
    if 'clean_sequences' not in dict_data:
        seqs = dict_data['sequences']
        clean_seqs = []
        for seq in seqs:
            s = str(seq).upper()
            clean_seqs.append("".join([c for c in s if c.isupper() and c.isalpha()]))
        dict_data['clean_sequences'] = clean_seqs
    return dict_data['clean_sequences']

def compute_aa_composition_clean(clean_sequences):
    comp_list = []
    for s_clean in clean_sequences:
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

def enrich_inputs_flexible(embeddings, sequences, clean_sequences, tmhmm_flags=None, ogt_priors=None,
                           use_aa_ratios=True, use_ogt_prior=True, use_tmhmm=True, use_seq_len=True):
    """Wrapper around enrich_inputs with feature toggle support."""
    N = embeddings.shape[0]
    parts = []
    
    # OGT prior (only for Tm head)
    if use_ogt_prior and ogt_priors is not None:
        ogt_t = torch.tensor([float(o) / 100.0 for o in ogt_priors], dtype=torch.float32).unsqueeze(-1)
        parts.append(ogt_t)
    
    # TMHMM flag
    if use_tmhmm:
        if tmhmm_flags is not None:
            tm_t = torch.tensor([float(f) for f in tmhmm_flags], dtype=torch.float32).unsqueeze(-1)
        else:
            tm_t = torch.zeros((N, 1), dtype=torch.float32)
        parts.append(tm_t)
    
    # Sequence length
    if use_seq_len:
        lengths = []
        for s_clean in clean_sequences:
            lengths.append(min(len(s_clean) / 2048.0, 1.0))
        parts.append(torch.tensor(lengths, dtype=torch.float32).unsqueeze(-1))
    
    # AA composition
    if use_aa_ratios:
        aa_comp = compute_aa_composition_clean(clean_sequences)
        parts.append(aa_comp)
    
    aux = torch.cat(parts, dim=-1) if parts else torch.zeros((N, 0), dtype=torch.float32)
    return embeddings.float(), aux

class AttentionPooling(nn.Module):
    def __init__(self, emb_dim=1280):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(emb_dim, 128), nn.Tanh(), nn.Linear(128, 1))
    def forward(self, x):
        weights = F.softmax(self.attn(x), dim=1)
        return torch.sum(x * weights, dim=1)

class CalibrationSaProtV8(nn.Module):
    def __init__(self, emb_dim=1280, aux_dim_tm=9, aux_dim_ogt=8, proj_dim=64,
                 hidden1=512, hidden2=256, mlp_layers=3, dropout1=0.3, dropout2=0.2,
                 norm_type='layernorm', use_residuals=True, loss_tm_type='nll_softplus',
                 pooling_method='mean', backbone_type='saprot_650m_last', config_dict=None):
        super().__init__()
        self.proj_dim = proj_dim
        self.mlp_layers = mlp_layers
        self.use_residuals = use_residuals
        self.loss_tm_type = loss_tm_type
        self.pooling_method = pooling_method
        self.backbone_type = backbone_type
        self.cfg = config_dict if config_dict is not None else BASE_CONFIG
        
        if 'layer30' in backbone_type or 'layer33' in backbone_type:
            self.backbone_adapt = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.LayerNorm(emb_dim), nn.GELU())
        else:
            self.backbone_adapt = None

        if pooling_method == 'attention':
            self.pool_layer = AttentionPooling(emb_dim)
        else:
            self.pool_layer = None
        
        if proj_dim is not None and proj_dim > 0:
            self.aux_proj_tm = nn.Sequential(nn.Linear(aux_dim_tm, proj_dim), nn.GELU(), nn.LayerNorm(proj_dim))
            self.aux_proj_ogt = nn.Sequential(nn.Linear(aux_dim_ogt, proj_dim), nn.GELU(), nn.LayerNorm(proj_dim))
            in_dim_tm = emb_dim + proj_dim
            in_dim_ogt = emb_dim + proj_dim
        else:
            self.aux_proj_tm = None
            self.aux_proj_ogt = None
            in_dim_tm = emb_dim + aux_dim_tm
            in_dim_ogt = emb_dim + aux_dim_ogt

        def build_mlp(in_dim, out_dim):
            layers = nn.ModuleList()
            norms = nn.ModuleList()
            res_projs = nn.ModuleList()
            curr_in = in_dim
            for l in range(mlp_layers):
                curr_out = hidden1 if l == 0 else hidden2
                layers.append(nn.Linear(curr_in, curr_out))
                norms.append(nn.BatchNorm1d(curr_out) if norm_type == 'batchnorm' else nn.LayerNorm(curr_out))
                res_projs.append(nn.Linear(curr_in, curr_out) if use_residuals and curr_in != curr_out else None)
                curr_in = curr_out
            head = nn.Linear(curr_in, out_dim)
            return layers, norms, res_projs, head

        out_dim_tm = 3 if loss_tm_type == 'quantile' else (2 if loss_tm_type == 'nll_softplus' else 1)
        self.tm_layers, self.tm_norms, self.tm_res, self.head_tm = build_mlp(in_dim_tm, out_dim_tm)
        self.ogt_layers, self.ogt_norms, self.ogt_res, self.head_ogt = build_mlp(in_dim_ogt, 1)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout1)
        self.drop2 = nn.Dropout(dropout2)

    def forward(self, x_emb, x_aux=None, head='tm'):
        if x_aux is None:
            x_aux = x_emb[:, -9:] if head == 'tm' else x_emb[:, -8:]
            x_emb = x_emb[:, :1280]

        if self.backbone_adapt is not None:
            x_emb = self.backbone_adapt(x_emb)

        if self.pool_layer is not None:
            x_emb = self.pool_layer(x_emb.unsqueeze(1))

        if head == 'tm':
            h_aux = self.aux_proj_tm(x_aux) if self.aux_proj_tm is not None else x_aux
            h = torch.cat([x_emb, h_aux], dim=-1)
            for l in range(self.mlp_layers):
                h_next = self.drop1(self.act(self.tm_norms[l](self.tm_layers[l](h)))) if l == 0 else self.drop2(self.act(self.tm_norms[l](self.tm_layers[l](h))))
                h = h_next + (h if self.tm_res[l] is None else self.tm_res[l](h)) if self.use_residuals else h_next
            out = self.head_tm(h)
            if self.loss_tm_type == 'nll_softplus':
                return out[:, 0], F.softplus(out[:, 1]) + self.cfg.get('nll_softplus_eps', 1e-4)
            elif self.loss_tm_type == 'quantile':
                return out
            else:
                return out.squeeze(-1)
        else:
            h_aux = self.aux_proj_ogt(x_aux) if self.aux_proj_ogt is not None else x_aux
            h = torch.cat([x_emb, h_aux], dim=-1)
            for l in range(self.mlp_layers):
                h_next = self.drop1(self.act(self.ogt_norms[l](self.ogt_layers[l](h)))) if l == 0 else self.drop2(self.act(self.ogt_norms[l](self.ogt_layers[l](h))))
                h = h_next + (h if self.ogt_res[l] is None else self.ogt_res[l](h)) if self.use_residuals else h_next
            return self.head_ogt(h).squeeze(-1)

def sanitize_and_weight_flexible(dict_data, is_tm=True, seq_len_min=50, filter_mode='remove'):
    seqs = dict_data['sequences']
    clean_seqs = get_clean_sequences(dict_data)
    embs = dict_data['embeddings']
    if is_tm:
        lbls = dict_data.get('tm_consensus', dict_data.get('labels'))
        ogts = dict_data.get('ogt')
        if ogts is None:
            ogts = torch.full((len(seqs),), 50.0)
    else:
        lbls = dict_data.get('ogt_consensus', dict_data.get('labels'))
        ogts = lbls
        
    keep_indices = []
    soft_weights = []
    
    lbls_is_tensor = isinstance(lbls, torch.Tensor)
    ogts_is_tensor = isinstance(ogts, torch.Tensor)
    
    lbls_np = lbls.numpy() if lbls_is_tensor else np.array(lbls, dtype=np.float32)
    ogts_np = ogts.numpy() if ogts_is_tensor else (np.array(ogts, dtype=np.float32) if ogts is not None else None)

    for idx, seq in enumerate(seqs):
        s_clean = clean_seqs[idx]
        if len(s_clean) < seq_len_min:
            continue
        w_mult = 1.0
        if is_tm and ogts_np is not None and idx < len(ogts_np):
            o_val = ogts_np[idx]
            if np.isnan(o_val):
                o_val = 0.0
            if lbls_np[idx] < o_val:
                if str(filter_mode).lower() in ['true', 'remove', '1']:
                    continue
                elif str(filter_mode).lower() == 'downweight_0.5':
                    w_mult = 0.5
                elif str(filter_mode).lower() == 'downweight_0.3':
                    w_mult = 0.3
        keep_indices.append(idx)
        soft_weights.append(w_mult)
        
    embs_sub = embs[keep_indices]
    seqs_sub = [seqs[i] for i in keep_indices]
    clean_seqs_sub = [clean_seqs[i] for i in keep_indices]
    
    if lbls_is_tensor:
        lbls_sub = lbls[keep_indices].clone().float()
    else:
        lbls_sub = torch.tensor(lbls_np[keep_indices], dtype=torch.float32)
        
    tmhmm_arr = dict_data.get('tmhmm_tm_binary')
    if tmhmm_arr is not None:
        if isinstance(tmhmm_arr, torch.Tensor):
            tmhmm_sub = tmhmm_arr[keep_indices].clone().int().tolist()
        else:
            tmhmm_sub = [tmhmm_arr[i] for i in keep_indices]
    else:
        tmhmm_sub = [0] * len(keep_indices)
        
    soft_weights_t = torch.tensor(soft_weights, dtype=torch.float32)
    
    if is_tm:
        if ogts_is_tensor:
            ogts_sub = ogts[keep_indices].clone().float().tolist()
        else:
            ogts_sub = [ogts[i] for i in keep_indices]
        return embs_sub, seqs_sub, clean_seqs_sub, lbls_sub, ogts_sub, tmhmm_sub, soft_weights_t
    else:
        return embs_sub, seqs_sub, clean_seqs_sub, lbls_sub, tmhmm_sub, soft_weights_t

def pinball_loss(preds, target, quantiles=[0.1, 0.5, 0.9], weights=None):
    loss = 0.0
    for i, q in enumerate(quantiles):
        err = target - preds[:, i]
        q_loss = torch.max(q * err, (q - 1) * err)
        if weights is not None:
            q_loss = q_loss * weights
        loss = loss + q_loss.mean()
    return loss / len(quantiles)

def parse_val(val_str):
    if str(val_str).lower() == 'true': return True
    if str(val_str).lower() == 'false': return False
    if str(val_str).lower() == 'none': return None
    try:
        if '.' in str(val_str) or 'e' in str(val_str).lower():
            return float(val_str)
        return int(val_str)
    except ValueError:
        return val_str

def get_cached_or_compute_enrichment(data, split_key, is_tm, seq_len_min, filter_mode, iqr_max, preloaded_data, cfg):
    use_aa_ratios = cfg.get('use_aa_ratios', True)
    use_ogt_prior = cfg.get('use_ogt_prior', True)
    use_tmhmm = cfg.get('use_tmhmm', True)
    use_seq_len = cfg.get('use_seq_len', True)

    if preloaded_data is not None:
        if 'cache_enrich' not in preloaded_data:
            preloaded_data['cache_enrich'] = {}
        cache = preloaded_data['cache_enrich']
        cache_key = (
            split_key,
            is_tm,
            seq_len_min,
            filter_mode,
            iqr_max if is_tm else None,
            use_aa_ratios,
            use_ogt_prior if is_tm else False,
            use_tmhmm,
            use_seq_len
        )
        if cache_key in cache:
            return cache[cache_key]

    t0 = time.time()
    if is_tm:
        emb, seq, clean_seq, lbl, ogt, tmhmm, soft = sanitize_and_weight_flexible(data[split_key], True, seq_len_min, filter_mode)
        if split_key == 'train_tm' and iqr_max is not None:
            iqr_map = get_seq_iqr_map()
            keep = []
            for idx, s in enumerate(seq):
                if str(s) in iqr_map and iqr_map[str(s)] > iqr_max:
                    continue
                keep.append(idx)
            if len(keep) < len(seq):
                emb = emb[keep]
                seq = [seq[i] for i in keep]
                clean_seq = [clean_seq[i] for i in keep]
                lbl = lbl[keep]
                ogt = [ogt[i] for i in keep]
                tmhmm = [tmhmm[i] for i in keep]
                soft = soft[keep]
        e_in, aux_in = enrich_inputs_flexible(emb, seq, clean_seq, tmhmm, ogt, use_aa_ratios, use_ogt_prior, use_tmhmm, use_seq_len)
        iqr_w = build_tm_iqr_weights(seq) if split_key == 'train_tm' else torch.ones_like(lbl)
        res = (e_in, aux_in, lbl, ogt, tmhmm, soft, iqr_w)
    else:
        emb, seq, clean_seq, lbl, tmhmm, soft = sanitize_and_weight_flexible(data[split_key], False, seq_len_min, filter_mode)
        e_in, aux_in = enrich_inputs_flexible(emb, seq, clean_seq, tmhmm, None, use_aa_ratios, use_ogt_prior, use_tmhmm, use_seq_len)
        res = (e_in, aux_in, lbl, tmhmm, soft)

    if preloaded_data is not None:
        preloaded_data['cache_enrich'][cache_key] = res
        print(f"  [Cache Miss] Computed & cached {split_key} features in {time.time()-t0:.1f}s", flush=True)
    return res

def run_single_sweep(group, param, value, seed=1, epochs=15, patience=5, debug=False, preloaded_data=None):
    cfg = copy.deepcopy(BASE_CONFIG)
    parsed_val = parse_val(value)
    cfg[param] = parsed_val
    if param == 'ensemble_seeds':
        seed = int(parsed_val)
    set_all_seeds(seed)

    print(f"\n{'='*65}\nSWEEP RUN: Group [{group}] | {param} = {parsed_val} | Seed: {seed}\n{'='*65}", flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start_time = time.time()

    backbone_type = str(cfg.get('backbone_type', 'saprot_650m_last'))
    if preloaded_data is not None:
        if '1.3b' in backbone_type and 'saprot_1.3b' in preloaded_data:
            data = preloaded_data['saprot_1.3b']
        else:
            data = preloaded_data['saprot_650m']
    else:
        if '1.3b' in backbone_type and os.path.exists("data/embeddings/prepared_data_v7_saprot1.3b_seqonly.pt"):
            data = torch.load("data/embeddings/prepared_data_v7_saprot1.3b_seqonly.pt", map_location='cpu', weights_only=False)
        else:
            data = torch.load("data/embeddings/saprot_tm_struct_embeddings.pt", map_location='cpu', weights_only=False)
        ogt_split = torch.load("data/embeddings/prepared_data_v7_saprot1.3b_seqonly_ogt_split.pt", map_location='cpu', weights_only=False)
        data['val_ogt'] = ogt_split['val_ogt']

    if debug:
        data = copy.deepcopy(data)
        for k in ['embeddings', 'sequences', 'tm_consensus', 'ogt', 'tmhmm_tm_binary', 'ids']:
            if k in data['train_tm']: data['train_tm'][k] = data['train_tm'][k][:200]
            if k in data['val_tm']: data['val_tm'][k] = data['val_tm'][k][:50]
        for k in ['embeddings', 'sequences', 'ogt_consensus', 'ogt_original', 'tmhmm_tm_binary', 'ids']:
            if k in data['train_ogt']: data['train_ogt'][k] = data['train_ogt'][k][:1000]
            if k in data['val_ogt']: data['val_ogt'][k] = data['val_ogt'][k][:200]

    # Set iqr scale configuration in imported train module
    import train
    train.CONFIG['iqr_scale'] = cfg.get('iqr_scale', 6.34)
    train.CONFIG['iqr_impute_val'] = cfg.get('iqr_impute_val', 0.62)

    seq_len_min = cfg.get('seq_len_min', 50)
    filter_mode = cfg.get('filter_tm_lt_ogt', 'remove')
    iqr_max = cfg.get('iqr_filter_max', 2.0)

    tr_tm_e, tr_tm_aux, tr_tm_lbl, tr_tm_ogt, tr_tm_tmhmm, tr_tm_soft, tm_iqr_weights = get_cached_or_compute_enrichment(data, 'train_tm', True, seq_len_min, filter_mode, iqr_max, preloaded_data, cfg)
    val_tm_e, val_tm_aux, val_tm_lbl, val_tm_ogt, val_tm_tmhmm, _, _ = get_cached_or_compute_enrichment(data, 'val_tm', True, seq_len_min, filter_mode, iqr_max, preloaded_data, cfg)
    tr_ogt_e, tr_ogt_aux, tr_ogt_lbl, tr_ogt_tmhmm, tr_ogt_soft = get_cached_or_compute_enrichment(data, 'train_ogt', False, seq_len_min, filter_mode, iqr_max, preloaded_data, cfg)
    val_ogt_e, val_ogt_aux, val_ogt_lbl, val_ogt_tmhmm, _ = get_cached_or_compute_enrichment(data, 'val_ogt', False, seq_len_min, filter_mode, iqr_max, preloaded_data, cfg)

    tm_mean, tm_std = tr_tm_lbl.mean().item(), tr_tm_lbl.std().item()
    ogt_mean, ogt_std = tr_ogt_lbl.mean().item(), tr_ogt_lbl.std().item()

    weight_power = cfg.get('weight_power', 0.75)
    clamp_max = cfg.get('weight_clamp_max', 22.0)
    tr_tm_w = compute_bin_weights_calib(tr_tm_lbl, cfg['bin_edges'], cfg['weight_clamp_min'], clamp_max, weight_power) * tm_iqr_weights * tr_tm_soft
    val_tm_w = torch.ones_like(val_tm_lbl)
    tr_ogt_w = compute_bin_weights_calib(tr_ogt_lbl, cfg['bin_edges'], cfg['weight_clamp_min'], clamp_max, weight_power) * tr_ogt_soft
    val_ogt_w = torch.ones_like(val_ogt_lbl)

    tr_ogt_lbl_z = (tr_ogt_lbl - ogt_mean) / ogt_std if cfg.get('ogt_normalize', True) else tr_ogt_lbl

    aug_prob = cfg.get('augment_prob', 0.15)
    aug_noise = cfg.get('augment_noise_std', 0.02)
    train_tm_ds = ProteinDataset(tr_tm_e, tr_tm_aux, tr_tm_lbl, tr_tm_w, augment=(aug_prob > 0), augment_prob=aug_prob, augment_noise_std=aug_noise)
    val_tm_ds = ProteinDataset(val_tm_e, val_tm_aux, val_tm_lbl, val_tm_w)
    train_ogt_ds = ProteinDataset(tr_ogt_e, tr_ogt_aux, tr_ogt_lbl_z, tr_ogt_w, augment=(aug_prob > 0), augment_prob=aug_prob, augment_noise_std=aug_noise)
    val_ogt_ds = ProteinDataset(val_ogt_e, val_ogt_aux, val_ogt_lbl, val_ogt_w)

    ogt_subsampler = MesophilicSubsampler(tr_ogt_lbl, keep_rate=cfg.get('ogt_subsample_meso_rate', 0.14)) if cfg.get('ogt_subsample_meso_rate', 1.0) < 1.0 else None

    save_dir = os.path.join(PROJECT_ROOT, f"experiments/src/model_calibration/checkpoints/{group}/{param}_{value}_seed{seed}")
    os.makedirs(save_dir, exist_ok=True)
    torch.save({'tm_mean': tm_mean, 'tm_std': tm_std, 'ogt_mean': ogt_mean, 'ogt_std': ogt_std}, os.path.join(save_dir, 'normalization_stats.pt'))

    aux_dim_tm = 0
    if cfg.get('use_ogt_prior', True): aux_dim_tm += 1
    if cfg.get('use_tmhmm', True): aux_dim_tm += 1
    if cfg.get('use_seq_len', True): aux_dim_tm += 1
    if cfg.get('use_aa_ratios', True): aux_dim_tm += 6

    aux_dim_ogt = 0
    if cfg.get('use_tmhmm', True): aux_dim_ogt += 1
    if cfg.get('use_seq_len', True): aux_dim_ogt += 1
    if cfg.get('use_aa_ratios', True): aux_dim_ogt += 6

    loss_tm_type = str(cfg.get('loss_tm_type', 'nll_softplus'))
    model = CalibrationSaProtV8(
        aux_dim_tm=aux_dim_tm,
        aux_dim_ogt=aux_dim_ogt,
        proj_dim=cfg.get('proj_dim', 64),
        hidden1=cfg.get('hidden_size_1', 512),
        hidden2=cfg.get('hidden_size_2', 256),
        mlp_layers=cfg.get('mlp_layers', 3),
        dropout1=cfg.get('dropout_1', 0.3),
        dropout2=cfg.get('dropout_2', 0.2),
        norm_type=cfg.get('norm_type', 'layernorm'),
        use_residuals=cfg.get('use_residuals', True),
        loss_tm_type=loss_tm_type,
        pooling_method=cfg.get('pooling_method', 'mean'),
        backbone_type=backbone_type,
        config_dict=cfg
    ).to(device)

    opt_tm = optim.AdamW(model.parameters(), lr=cfg.get('learning_rate', 1e-4), weight_decay=cfg.get('weight_decay', 1e-5))
    opt_ogt = optim.AdamW(model.parameters(), lr=cfg.get('learning_rate', 1e-4), weight_decay=cfg.get('weight_decay', 1e-5))

    sched_tm = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_tm, T_0=10, T_mult=2, eta_min=1e-6)
    sched_ogt = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_ogt, T_0=10, T_mult=2, eta_min=1e-6)

    batch_sz = cfg.get('batch_size', 64)
    tm_loader = DataLoader(train_tm_ds, batch_size=batch_sz, shuffle=True, drop_last=True)
    ogt_loader = DataLoader(train_ogt_ds, batch_size=batch_sz, shuffle=True, drop_last=True)
    val_tm_loader = DataLoader(val_tm_ds, batch_size=batch_sz, shuffle=False)
    val_ogt_loader = DataLoader(val_ogt_ds, batch_size=batch_sz, shuffle=False)

    best_tm_mae = float('inf')
    best_ogt_mae = float('inf')
    patience_tm = 0
    patience_ogt = 0
    path_tm = os.path.join(save_dir, 'model_tm.pt')
    path_ogt = os.path.join(save_dir, 'model_ogt.pt')
    warmup_epochs = cfg.get('warmup_epochs', 0)

    for epoch in range(epochs):
        model.train()
        if epoch < warmup_epochs and warmup_epochs > 0:
            factor = (epoch + 1) / float(warmup_epochs)
            for pg in opt_tm.param_groups: pg['lr'] = cfg.get('learning_rate', 1e-4) * factor
            for pg in opt_ogt.param_groups: pg['lr'] = cfg.get('learning_rate', 1e-4) * factor

        if ogt_subsampler is not None:
            epoch_idx = ogt_subsampler.sample_epoch_indices()
            epoch_ogt_ds = torch.utils.data.Subset(train_ogt_ds, epoch_idx)
            ogt_loader = DataLoader(epoch_ogt_ds, batch_size=batch_sz, shuffle=True, drop_last=True)

        tm_iter = iter(cycle(tm_loader))
        for emb_ogt, aux_ogt, y_ogt, w_ogt in ogt_loader:
            emb_ogt, aux_ogt, y_ogt, w_ogt = emb_ogt.to(device), aux_ogt.to(device), y_ogt.to(device), w_ogt.to(device)
            if cfg.get('tm_ogt_noise_std', 0.0) > 0:
                emb_ogt = emb_ogt + torch.randn_like(emb_ogt) * cfg['tm_ogt_noise_std'] * 0.01

            if cfg.get('target_jitter_std', 0.5) > 0:
                y_ogt_raw = y_ogt * ogt_std + ogt_mean + torch.randn_like(y_ogt) * cfg['target_jitter_std']
                y_ogt = (y_ogt_raw - ogt_mean) / ogt_std

            opt_ogt.zero_grad()
            pred_o = model(emb_ogt, aux_ogt, head='ogt')
            loss_ogt = focal_huber_loss(pred_o, y_ogt, delta=cfg.get('huber_delta_ogt', 15.0)/ogt_std,
                                        gamma=cfg.get('focal_gamma', 2.0), beta=cfg.get('focal_beta', 0.5), weights=w_ogt)
            loss_ogt.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.get('grad_clip_max_norm', 1.0))
            opt_ogt.step()

            emb_tm, aux_tm, y_tm, w_tm = next(tm_iter)
            emb_tm, aux_tm, y_tm, w_tm = emb_tm.to(device), aux_tm.to(device), y_tm.to(device), w_tm.to(device)

            mixup_alpha = cfg.get('mixup_alpha', 0.0)
            if mixup_alpha > 0:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                idx_perm = torch.randperm(emb_tm.size(0))
                emb_tm = lam * emb_tm + (1 - lam) * emb_tm[idx_perm]
                aux_tm = lam * aux_tm + (1 - lam) * aux_tm[idx_perm]
                y_tm = lam * y_tm + (1 - lam) * y_tm[idx_perm]
                w_tm = lam * w_tm + (1 - lam) * w_tm[idx_perm]

            if cfg.get('tm_ogt_noise_std', 0.0) > 0:
                emb_tm = emb_tm + torch.randn_like(emb_tm) * cfg['tm_ogt_noise_std'] * 0.01
            z_tm = (y_tm - tm_mean) / tm_std

            opt_tm.zero_grad()
            out_tm = model(emb_tm, aux_tm, head='tm')
            if loss_tm_type == 'nll_softplus':
                z_mu, z_var = out_tm
                loss_tm = (0.5 * (z_mu - z_tm)**2 / z_var + 0.5 * torch.log(z_var)) * w_tm
                loss_tm = loss_tm.mean()
            elif loss_tm_type == 'quantile':
                loss_tm = pinball_loss(out_tm, z_tm, quantiles=[0.1, 0.5, 0.9], weights=w_tm)
            elif loss_tm_type == 'huber_5':
                loss_tm = (F.huber_loss(out_tm, z_tm, delta=5.0/tm_std, reduction='none') * w_tm).mean()
            elif loss_tm_type == 'huber_10':
                loss_tm = (F.huber_loss(out_tm, z_tm, delta=10.0/tm_std, reduction='none') * w_tm).mean()
            elif loss_tm_type == 'mse':
                loss_tm = (F.mse_loss(out_tm, z_tm, reduction='none') * w_tm).mean()
            else:
                loss_tm = (F.l1_loss(out_tm, z_tm, reduction='none') * w_tm).mean()

            loss_tm.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.get('grad_clip_max_norm', 1.0))
            opt_tm.step()

        model.eval()
        val_tm_mae = 0.0
        with torch.no_grad():
            for emb, aux, y, _ in val_tm_loader:
                emb, aux, y = emb.to(device), aux.to(device), y.to(device)
                out_tm = model(emb, aux, head='tm')
                if loss_tm_type == 'nll_softplus':
                    pred_z = out_tm[0]
                elif loss_tm_type == 'quantile':
                    pred_z = out_tm[:, 1]
                else:
                    pred_z = out_tm
                val_tm_mae += torch.abs((pred_z * tm_std + tm_mean) - y).sum().item()
        val_tm_mae /= len(val_tm_loader.dataset)

        val_ogt_mae = 0.0
        with torch.no_grad():
            for emb, aux, y, _ in val_ogt_loader:
                emb, aux, y = emb.to(device), aux.to(device), y.to(device)
                pred_o = model(emb, aux, head='ogt')
                val_ogt_mae += torch.abs((pred_o * ogt_std + ogt_mean) - y).sum().item()
        val_ogt_mae /= len(val_ogt_loader.dataset)

        if epoch >= warmup_epochs or warmup_epochs == 0:
            sched_tm.step()
            sched_ogt.step()

        if val_tm_mae < best_tm_mae:
            best_tm_mae = val_tm_mae
            patience_tm = 0
            torch.save(model.state_dict(), path_tm)
        else:
            patience_tm += 1

        if val_ogt_mae < best_ogt_mae:
            best_ogt_mae = val_ogt_mae
            patience_ogt = 0
            torch.save(model.state_dict(), path_ogt)
        else:
            patience_ogt += 1

        if patience_tm >= patience and patience_ogt >= patience:
            break

    elapsed_s = time.time() - start_time
    print(f"-> Run Complete | Best Val Tm MAE: {best_tm_mae:.4f}°C | Best Val OGT MAE: {best_ogt_mae:.4f}°C | Time: {elapsed_s:.1f}s", flush=True)

    csv_dir = os.path.join(PROJECT_ROOT, "experiments/src/model_calibration/logs")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, "summary_metrics.csv")
    file_exists = os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['group', 'param', 'value', 'seed', 'epochs_run', 'best_val_tm_mae', 'best_val_ogt_mae', 'runtime_sec'])
        writer.writerow([group, param, value, seed, epoch+1, f"{best_tm_mae:.4f}", f"{best_ogt_mae:.4f}", f"{elapsed_s:.1f}"])
    return best_tm_mae, best_ogt_mae

def main():
    parser = argparse.ArgumentParser(description="Isolated Model Calibration Sweep Runner")
    parser.add_argument("--group", type=str, required=True, help="Sweep group name")
    parser.add_argument("--param", type=str, required=True, help="Parameter name to override")
    parser.add_argument("--value", type=str, required=True, help="Value to test")
    parser.add_argument("--seed", type=int, default=1, help="Random seed (default: 1)")
    parser.add_argument("--epochs", type=int, default=15, help="Max epochs (default: 15)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (default: 5)")
    parser.add_argument("--debug", action="store_true", help="Run 1% slice for rapid verification")
    args = parser.parse_args()

    run_single_sweep(args.group, args.param, args.value, args.seed, args.epochs, args.patience, args.debug)

if __name__ == "__main__":
    main()
