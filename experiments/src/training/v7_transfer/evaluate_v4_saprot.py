import os
import csv
import json
import torch
import torch.nn as nn
import numpy as np
import scipy.stats
from config import CONFIG
from model import StableProtV7

def compute_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    pcc, _ = scipy.stats.pearsonr(y_true, y_pred)
    spearman, _ = scipy.stats.spearmanr(y_true, y_pred)
    r2 = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
    
    y_true_b = (y_true >= 60.0).astype(int)
    y_pred_b = (y_pred >= 60.0).astype(int)
    
    tp = np.sum((y_true_b == 1) & (y_pred_b == 1))
    fp = np.sum((y_true_b == 0) & (y_pred_b == 1))
    fn = np.sum((y_true_b == 1) & (y_pred_b == 0))
    tn = np.sum((y_true_b == 0) & (y_pred_b == 0))
    
    acc = (tp + tn) / len(y_true_b) if len(y_true_b) > 0 else 0.0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    
    denom = np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    mcc = (float(tp * tn) - float(fp * fn)) / denom if denom > 0 else 0.0
    
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1.0))) * 100.0
    
    from sklearn.metrics import roc_auc_score as _roc_auc
    global_aucs = []
    for t in range(30, 91):
        y_bin = (y_true >= t).astype(int)
        if len(np.unique(y_bin)) > 1:
            try:
                global_aucs.append(_roc_auc(y_bin, y_pred))
            except Exception:
                pass
    global_auc = np.mean(global_aucs) if global_aucs else 0.5
        
    return {
        'mae': mae, 'rmse': rmse, 'pcc': pcc, 'spearman': spearman, 'r2': r2,
        'acc': acc, 'sens': sens, 'spec': spec, 'prec': prec, 'f1': f1, 'mcc': mcc,
        'mape': mape, 'global_auc': global_auc
    }

def map_fireprot_to_uniprot_ids(fp_seqs, base_dir):
    sql_path = "/home/bibhu/Documents/temstampto/data/training_data/raw/fireprotdb_dump_2025_09_22/01_fireprotdb_2025-09-20.sql"
    csv_path = "/home/bibhu/Documents/temstampto/data/training_data/raw/fireprotdb_dump_2025_09_22/fireprotdb_csv_whole/fireprotdb_20251015-164116.csv"
    
    sequences = {}
    in_copy_block = False
    with open(sql_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("COPY public.sequence "):
                in_copy_block = True
                continue
            if in_copy_block:
                if line.strip() == "\\.":
                    break
                parts = line.split("\t")
                if len(parts) >= 2:
                    sequences[parts[0].strip()] = parts[1].strip().upper()
                    
    seq_to_id = {seq: seq_id for seq_id, seq in sequences.items()}
    
    seq_id_to_uniprot = {}
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) > 38:
                seq_id_to_uniprot[row[1].strip()] = row[38].strip()
                
    uids = []
    for seq in fp_seqs:
        seq_id = seq_to_id.get(seq)
        if seq_id and seq_id in seq_id_to_uniprot:
            uids.append(seq_id_to_uniprot[seq_id])
        else:
            uids.append(None)
    return uids

def evaluate_mode(mode, device, base_dir, x_saprot, y_true, uids, ogt_lookup, tm_lookup):
    results_dir = os.path.join(base_dir, 'results/saprot')
    use_ogt_feature = (mode in ['D3', 'D4'])
    use_tm_feature = (mode == 'D4')
    
    ensemble_preds = []
    saprot_input_dim = 1280
    
    for seed in CONFIG['seeds']:
        model_path = os.path.join(results_dir, f'model_{mode}.pt')
        # Since Stage 1 SaProt model is not available/needed (we train Stage 2 scratch),
        # we do not load a Stage 1 predictor. Fallback to lookup or 37.0.
        
        if not os.path.exists(model_path):
            continue
            
        model = StableProtV7(
            emb_dim=saprot_input_dim,
            hidden=CONFIG['hidden_size'],
            bottleneck=CONFIG['bottleneck_size'],
            use_ogt_feature=use_ogt_feature,
            use_tm_feature=use_tm_feature
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        
        ogt_vals = []
        tm_vals = []
        
        for i, uid in enumerate(uids):
            tm_val = float(tm_lookup.get(uid, 0.0)) if uid else 0.0
            tm_vals.append(tm_val)
            
            ogt_val = None
            if uid:
                ogt_info = ogt_lookup.get(uid, {})
                if ogt_info.get("source") == "known" and "ogt" in ogt_info:
                    ogt_val = float(ogt_info["ogt"])
            
            if ogt_val is None:
                ogt_val = 37.0
                    
            ogt_vals.append(ogt_val)
            
        ogt_tensor = torch.tensor(ogt_vals, dtype=torch.float32).to(device) if use_ogt_feature else None
        tm_tensor = torch.tensor(tm_vals, dtype=torch.float32).to(device) if use_tm_feature else None
        
        with torch.no_grad():
            pred = model(x_saprot, stage='tm', ogt_pred=ogt_tensor, tm_feat=tm_tensor).cpu().numpy()
            
        ensemble_preds.append(pred)
        
    if not ensemble_preds:
        print(f"No trained models found for Mode {mode}")
        return None
        
    mean_preds = np.mean(ensemble_preds, axis=0)
    metrics = compute_metrics(y_true, mean_preds)
    
    boot_maes, boot_pccs = [], []
    np.random.seed(42)
    n = len(y_true)
    for _ in range(1000):
        idx = np.random.choice(n, size=n, replace=True)
        boot_maes.append(np.mean(np.abs(y_true[idx] - mean_preds[idx])))
        pcc, _ = scipy.stats.pearsonr(y_true[idx], mean_preds[idx])
        boot_pccs.append(pcc)
        
    metrics['mae_ci'] = (np.percentile(boot_maes, 2.5), np.percentile(boot_maes, 97.5))
    metrics['pcc_ci'] = (np.percentile(boot_pccs, 2.5), np.percentile(boot_pccs, 97.5))
    
    return mean_preds, metrics

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = os.path.dirname(os.path.abspath(__file__))
    holdout_path = "/home/bibhu/Documents/temstampto/data/test_data/fireprot_holdout_saprot.pt"
    ogt_lookup_path = "/home/bibhu/Documents/temstampto/data/cleaner_data/tm_ogt_lookup.json"
    tm_lookup_path = "/home/bibhu/Documents/temstampto/data/cleaner_data/tm_transmembrane.json"
    
    if not os.path.exists(holdout_path):
        print(f"CRITICAL ERROR: SaProt holdout dataset missing at {holdout_path}")
        return
        
    print("Loading FireProt SaProt holdout dataset...")
    d_fireprot = torch.load(holdout_path, map_location="cpu")
    x_saprot = d_fireprot['embeddings_saprot'].to(device)
    y_true = d_fireprot['temperatures'].numpy() if hasattr(d_fireprot['temperatures'], 'numpy') else np.array(d_fireprot['temperatures'])
    fp_seqs = d_fireprot['sequences']
    
    print(f"FireProt Holdout: {len(y_true)} testing targets.")
    
    print("Mapping FireProt sequences to UniProtKB IDs...")
    uids = map_fireprot_to_uniprot_ids(fp_seqs, base_dir)
    
    print("Loading feature lookup files...")
    with open(ogt_lookup_path) as f:
        ogt_lookup = json.load(f)
    with open(tm_lookup_path) as f:
        tm_lookup = json.load(f)
        
    modes = ['D1', 'D2', 'D3', 'D4']
    results = {}
    
    print("\nEvaluating SaProt V7 Model Variants (D1-D4)...")
    for mode in modes:
        res = evaluate_mode(mode, device, base_dir, x_saprot, y_true, uids, ogt_lookup, tm_lookup)
        if res is not None:
            preds, metrics = res
            results[mode] = {'preds': preds, 'metrics': metrics}
            
    if not results:
        print("No evaluation results generated.")
        return
        
    print("\n" + "="*120)
    print(f"{'SaProt Model Variant':<25} | {'MAE (°C)':<15} | {'MAE 95% CI':<18} | {'PCC':<8} | {'PCC 95% CI':<18} | {'R²':<8} | {'Global AUC':<10}")
    print("="*120)
    
    for mode, data in results.items():
        m = data['metrics']
        mae_str = f"{m['mae']:.2f}"
        mae_ci_str = f"[{m['mae_ci'][0]:.2f}, {m['mae_ci'][1]:.2f}]"
        pcc_str = f"{m['pcc']:.2f}"
        pcc_ci_str = f"[{m['pcc_ci'][0]:.2f}, {m['pcc_ci'][1]:.2f}]"
        r2_str = f"{m['r2']:.2f}"
        auc_str = f"{m['global_auc']:.3f}"
        
        mode_name = f"SaProt Mode {mode}"
        if mode == 'D1':
            mode_name += " (Baseline Clean)"
        elif mode == 'D2':
            mode_name += " (+Stratified)"
        elif mode == 'D3':
            mode_name += " (+OGT lookup)"
        elif mode == 'D4':
            mode_name += " (+TM lookup)"
            
        print(f"{mode_name:<25} | {mae_str:<15} | {mae_ci_str:<18} | {pcc_str:<8} | {pcc_ci_str:<18} | {r2_str:<8} | {auc_str:<10}")
    print("="*120)

    out_path = os.path.join(base_dir, 'v7_fireprot_eval_metrics_saprot.json')
    serializable_results = {}
    for mode, data in results.items():
        m = data['metrics']
        serializable_results[mode] = {
            'mae': float(m['mae']),
            'mae_ci': [float(m['mae_ci'][0]), float(m['mae_ci'][1])],
            'pcc': float(m['pcc']),
            'pcc_ci': [float(m['pcc_ci'][0]), float(m['pcc_ci'][1])],
            'r2': float(m['r2']),
            'global_auc': float(m['global_auc']),
            'f1': float(m['f1']),
            'mcc': float(m['mcc'])
        }
    with open(out_path, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    print(f"Metrics saved to {out_path}")

if __name__ == "__main__":
    main()
