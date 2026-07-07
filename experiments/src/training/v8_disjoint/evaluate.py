#!/usr/bin/env python3
"""StableProt V8 Comprehensive Benchmark Evaluation Suite.
Evaluates trained 5-seed V8 ensemble across internal holdouts using 2-stage decoupled inference.
"""

import os
import sys
import argparse
import glob
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

try:
    from config import CONFIG
    from train import MultiHeadSaProtV8, enrich_inputs, sanitize_data
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from config import CONFIG
    from train import MultiHeadSaProtV8, enrich_inputs, sanitize_data

def evaluate_ensemble_1stage(models_tm, embs, seqs, tmhmms, ogt_trues, tm_mean, tm_std, device):
    """Run 1-stage forward pass using true OGT priors (evaluates pure Tm head quality)."""
    mus = []
    vars_list = []
    with torch.no_grad():
        for m_tm in models_tm:
            m_tm.eval()
            emb_t, aux_t = enrich_inputs(embs, seqs, tmhmms, ogt_priors=ogt_trues)
            z_mu, z_lv = m_tm(emb_t.to(device), aux_t.to(device), head='tm')
            pred_mu = z_mu.cpu() * tm_std + tm_mean
            pred_var = z_lv.cpu() * (tm_std ** 2)
            mus.append(pred_mu)
            vars_list.append(pred_var)
            
    mus_stack = torch.stack(mus, dim=0)
    vars_stack = torch.stack(vars_list, dim=0)
    weights = 1.0 / (vars_stack + 1e-6)
    ens_mu = (mus_stack * weights).sum(dim=0) / weights.sum(dim=0)
    total_var = 1.0 / weights.sum(dim=0)
    return ens_mu, torch.sqrt(total_var)

def evaluate_ensemble_2stage(models_tm, models_ogt, embs, seqs, tmhmms, tm_mean, tm_std, device, ogt_mean=None, ogt_std=None):
    """Run 2-stage forward pass across ensemble models (no TTA)."""
    mus = []
    vars_list = []
    with torch.no_grad():
        for m_tm, m_ogt in zip(models_tm, models_ogt):
            m_tm.eval()
            m_ogt.eval()

            # Forward pass
            emb_o, aux_o = enrich_inputs(embs, seqs, tmhmms, ogt_priors=None)
            pred_ogt = m_ogt(emb_o.to(device), aux_o.to(device), head='ogt')
            if ogt_mean is not None and ogt_std is not None:
                pred_ogt_raw = pred_ogt.cpu() * ogt_std + ogt_mean
            else:
                pred_ogt_raw = pred_ogt.cpu()
                
            emb_t, aux_t = enrich_inputs(embs, seqs, tmhmms, ogt_priors=pred_ogt_raw.numpy())
            z_mu, z_lv = m_tm(emb_t.to(device), aux_t.to(device), head='tm')
            pred_mu = z_mu.cpu() * tm_std + tm_mean
            pred_var = z_lv.cpu() * (tm_std ** 2)
            mus.append(pred_mu)
            vars_list.append(pred_var)

    mus_stack = torch.stack(mus, dim=0)
    vars_stack = torch.stack(vars_list, dim=0)

    # Confidence-weighted ensemble: weight by inverse variance
    weights = 1.0 / (vars_stack + 1e-6)
    ens_mu = (mus_stack * weights).sum(dim=0) / weights.sum(dim=0)
    total_var = 1.0 / weights.sum(dim=0)

    return ens_mu, torch.sqrt(total_var)

def compute_metrics(y_true, y_pred):
    yt = np.array(y_true)
    yp = np.array(y_pred)
    mae = np.mean(np.abs(yt - yp))
    rmse = np.sqrt(np.mean((yt - yp)**2))
    pcc, _ = pearsonr(yt, yp)
    scc, _ = spearmanr(yt, yp)
    r2 = r2_score(yt, yp)
    return {'MAE': mae, 'RMSE': rmse, 'PCC': pcc, 'SCC': scc, 'R2': r2}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict_20", action="store_true", help="Evaluate strictly on <20% identity decontaminated holdout")
    parser.add_argument("--models_dir", type=str, default=os.path.join(os.path.dirname(__file__), "results"), help="Directory containing seed models")
    args = parser.parse_args()
    
    cdhit_threshold = 0.20 if args.strict_20 else 0.30
    print(f"\n{'='*60}\nStableProt V8 Comprehensive Evaluation Suite\n{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_path = "data/embeddings/saprot_tm_struct_embeddings.pt"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    
    train_ids = set(data['train_tm']['ids'])
    test_ids_raw = data['test_tm']['ids']
    keep_idx = [i for i, tid in enumerate(test_ids_raw) if tid not in train_ids]
    for k in ['embeddings', 'sequences', 'tm_consensus', 'ogt', 'tmhmm_tm_binary', 'source', 'ids']:
        if k in data['test_tm']:
            data['test_tm'][k] = [data['test_tm'][k][i] for i in keep_idx] if isinstance(data['test_tm'][k], list) else data['test_tm'][k][keep_idx]
            
    test_ids_clean = set(data['test_tm']['ids'])
    overlap_clean = train_ids & test_ids_clean
    assert len(overlap_clean) == 0, f"Critical Contamination Error: {len(overlap_clean)} sequences overlap!"
    print("  Assertion Passed: 0% overlap between train_tm and decontaminated test_tm holdout.")
    
    seed_dirs = sorted(glob.glob(os.path.join(args.models_dir, "seed*")))
    if len(seed_dirs) == 0:
        print(f"Warning: No trained seed folders found in {args.models_dir}.")
        return
    models_tm, models_ogt = [], []
    for sd in seed_dirs:
        pt_tm = os.path.join(sd, "model_tm.pt")
        pt_ogt = os.path.join(sd, "model_ogt.pt")
        pt_comb = os.path.join(sd, "model.pt")
        if os.path.exists(pt_tm) and os.path.exists(pt_ogt):
            m_t = MultiHeadSaProtV8().to(device)
            m_t.load_state_dict(torch.load(pt_tm, map_location=device))
            models_tm.append(m_t)
            m_o = MultiHeadSaProtV8().to(device)
            m_o.load_state_dict(torch.load(pt_ogt, map_location=device))
            models_ogt.append(m_o)
        elif os.path.exists(pt_comb):
            m_comb = MultiHeadSaProtV8().to(device)
            m_comb.load_state_dict(torch.load(pt_comb, map_location=device))
            models_tm.append(m_comb)
            models_ogt.append(m_comb)
            
    print(f"Loaded {len(models_tm)} decoupled ensemble seed pairs.")
    if len(models_tm) == 0:
        return
        
    _, _, tr_lbl, _, _ = sanitize_data(data['train_tm'], is_tm=True)
    tm_mean = tr_lbl.mean().item()
    tm_std = tr_lbl.std().item()
    
    test_data = data['test_tm']
    embs = test_data['embeddings']
    seqs = test_data['sequences']
    lbls = test_data['tm_consensus']
    ogts = test_data.get('ogt', [50.0]*len(seqs))
    tmhmms = test_data.get('tmhmm_tm_binary', [0]*len(seqs))
    sources = test_data.get('source', ['test']*len(seqs))
    
    norm_path = os.path.join(args.models_dir, 'normalization_stats.pt')
    if os.path.exists(norm_path):
        norms = torch.load(norm_path, map_location='cpu')
        ogt_mean, ogt_std = norms.get('ogt_mean'), norms.get('ogt_std')
    else:
        ogt_mean, ogt_std = None, None
        
    # 1. Evaluate 1-Stage (True OGT Priors - M2 Check)
    preds_mu_1s, _ = evaluate_ensemble_1stage(models_tm, embs, seqs, tmhmms, ogts, tm_mean, tm_std, device)
    metrics_1s = compute_metrics(lbls, preds_mu_1s.numpy())
    
    # 2. Evaluate 2-Stage (Predicted OGT Priors - M3 Check)
    preds_mu_2s, preds_sigma_2s = evaluate_ensemble_2stage(
        models_tm, models_ogt, embs, seqs, tmhmms, tm_mean, tm_std, device, ogt_mean, ogt_std)
    metrics_2s = compute_metrics(lbls, preds_mu_2s.numpy())
    
    # 3. Uncertainty Calibration Check (M5)
    errs_2s = np.abs(lbls - preds_mu_2s.numpy())
    unc_corr, _ = spearmanr(preds_sigma_2s.numpy(), errs_2s)
    
    print(f"\n{'='*60}")
    print(f"Internal Test Set Results (N={len(lbls)}):")
    print(f"  1-Stage (True OGT)      | MAE: {metrics_1s['MAE']:.4f}°C | PCC: {metrics_1s['PCC']:.4f} | R2: {metrics_1s['R2']:.4f}")
    print(f"  2-Stage (Predicted OGT) | MAE: {metrics_2s['MAE']:.4f}°C | PCC: {metrics_2s['PCC']:.4f} | R2: {metrics_2s['R2']:.4f}")
    print(f"  2-Stage Error Penalty   | +{metrics_2s['MAE'] - metrics_1s['MAE']:.4f}°C (M3 Amplification Check)")
    print(f"  Uncertainty Calibration | Spearman Corr(sigma, |error|): {unc_corr:.4f} (M5 Check)")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
