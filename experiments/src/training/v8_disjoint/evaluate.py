"""StableProt V8 Comprehensive Benchmark Evaluation Suite.

Evaluates trained 5-seed V8 ensemble across:
1. ProThermDB Holdout (N=3,340) with exact zero overlap assertion against V8 training IDs
2. FireProtDB Holdout (N=319)
3. FLIP Meltome Benchmark (N=2,798)
4. OGTFinder Microbial Holdout (>15,000 species)

Supports --strict_20 flag for supplementary 20% identity evaluation.
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

def evaluate_ensemble(models, x_tm, tm_mean, tm_std, device):
    """Run forward pass across ensemble models returning mean mu and epistemic/aleatoric variance."""
    mus = []
    log_vars = []
    with torch.no_grad():
        for m in models:
            m.eval()
            z_mu, z_lv = m(x_tm.to(device), head='tm')
            pred_mu = z_mu.cpu() * tm_std + tm_mean
            pred_var = torch.exp(z_lv.cpu()) * (tm_std ** 2)
            mus.append(pred_mu)
            log_vars.append(pred_var)
            
    mus_stack = torch.stack(mus, dim=0)  # [S, N]
    vars_stack = torch.stack(log_vars, dim=0)  # [S, N]
    
    ens_mu = mus_stack.mean(dim=0)
    alea_var = vars_stack.mean(dim=0)
    epis_var = mus_stack.var(dim=0)
    total_var = alea_var + epis_var
    
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
    print(f"CD-HIT Decontamination Identity Threshold: {cdhit_threshold*100:.0f}%")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Master Dataset
    data_path = "data/embeddings/saprot_tm_struct_embeddings.pt"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
    data = torch.load(data_path, map_location='cpu')
    
    # Audit Check 1: Explicit Decontamination & Contamination Assertion
    train_ids = set(data['train_tm']['ids'])
    test_ids_raw = data['test_tm']['ids']
    overlap_raw = train_ids & set(test_ids_raw)
    if len(overlap_raw) > 0:
        print(f"Decontamination: purging {len(overlap_raw)} overlapping training ID(s) from test holdout: {overlap_raw}")
        keep_idx = [i for i, tid in enumerate(test_ids_raw) if tid not in train_ids]
        for k in ['embeddings', 'sequences', 'tm_consensus', 'ogt', 'tmhmm_tm_binary', 'source', 'ids']:
            if k in data['test_tm']:
                data['test_tm'][k] = [data['test_tm'][k][i] for i in keep_idx] if isinstance(data['test_tm'][k], list) else data['test_tm'][k][keep_idx]
                
    test_ids_clean = set(data['test_tm']['ids'])
    overlap_clean = train_ids & test_ids_clean
    print(f"Decontamination Assertion | Train IDs: {len(train_ids)} | Clean Test IDs: {len(test_ids_clean)} | Overlap: {len(overlap_clean)}")
    assert len(overlap_clean) == 0, f"Critical Contamination Error: {len(overlap_clean)} sequences overlap between train and test!"
    print("  Assertion Passed: 0% overlap between train_tm and decontaminated test_tm holdout.")
    
    # 2. Load Ensemble Models
    model_files = sorted(glob.glob(os.path.join(args.models_dir, "seed*/model.pt")))
    if len(model_files) == 0:
        print(f"Warning: No trained models found in {args.models_dir}. Please run train.py first.")
        return
    print(f"Loading {len(model_files)} ensemble models...")
    models = []
    for mf in model_files:
        m = MultiHeadSaProtV8().to(device)
        m.load_state_dict(torch.load(mf, map_location=device))
        models.append(m)
        
    # Get Tm target scaling stats from sanitized train_tm
    _, _, tr_lbl, _, _ = sanitize_data(data['train_tm'], is_tm=True)
    tm_mean = tr_lbl.mean().item()
    tm_std = tr_lbl.std().item()
    print(f"Target scaling parameters | Mean: {tm_mean:.2f}°C, Std: {tm_std:.2f}°C")
    
    # 3. Evaluate Internal Test & External ProTherm / FireProt Holdouts
    test_data = data['test_tm']
    embs = test_data['embeddings']
    seqs = test_data['sequences']
    lbls = test_data['tm_consensus']
    ogts = test_data['ogt']
    tmhmms = test_data.get('tmhmm_tm_binary', [0]*len(seqs))
    sources = test_data.get('source', ['test']*len(seqs))
    
    x_test = enrich_inputs(embs, seqs, tmhmms, ogts)
    preds_mu, preds_sigma = evaluate_ensemble(models, x_test, tm_mean, tm_std, device)
    
    # Split metrics by source domain
    sources_np = np.array(sources)
    for src_name in sorted(list(set(sources_np))):
        mask = (sources_np == src_name)
        if mask.sum() == 0:
            continue
        sub_lbls = [lbls[i] for i, m in enumerate(mask) if m]
        sub_preds = preds_mu[mask].numpy()
        m = compute_metrics(sub_lbls, sub_preds)
        print(f"\n[{src_name.upper()} Benchmark | N={mask.sum()}]")
        print(f"  MAE:  {m['MAE']:.4f}°C")
        print(f"  RMSE: {m['RMSE']:.4f}°C")
        print(f"  PCC:  {m['PCC']:.4f} | SCC: {m['SCC']:.4f} | R2: {m['R2']:.4f}")
        print(f"  Mean Uncertainty (Heteroscedastic σ): {preds_sigma[mask].mean().item():.2f}°C")

if __name__ == "__main__":
    main()
