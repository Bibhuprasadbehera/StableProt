import os
import json
import torch
import torch.nn as nn
import numpy as np
import scipy.stats
from config import CONFIG
from model import StableProtV7

def load_compat_state_dict(model, state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("backbone.0."):
            new_state_dict[k.replace("backbone.0.", "input_layer.")] = v
        elif k.startswith("backbone.1."):
            new_state_dict[k.replace("backbone.1.", "backbone_rest.0.")] = v
        elif k.startswith("backbone.2."):
            new_state_dict[k.replace("backbone.2.", "backbone_rest.1.")] = v
        elif k.startswith("backbone.3."):
            new_state_dict[k.replace("backbone.3.", "backbone_rest.2.")] = v
        elif k.startswith("backbone.4."):
            new_state_dict[k.replace("backbone.4.", "backbone_rest.3.")] = v
        elif k.startswith("backbone.5."):
            new_state_dict[k.replace("backbone.5.", "backbone_rest.4.")] = v
        elif k.startswith("backbone.6."):
            new_state_dict[k.replace("backbone.6.", "backbone_rest.5.")] = v
        elif k.startswith("backbone.7."):
            new_state_dict[k.replace("backbone.7.", "backbone_rest.6.")] = v
        elif k.startswith("backbone.8."):
            new_state_dict[k.replace("backbone.8.", "backbone_rest.7.")] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)

def compute_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    pcc, _ = scipy.stats.pearsonr(y_true, y_pred)
    spearman, _ = scipy.stats.spearmanr(y_true, y_pred)
    r2 = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
    
    # Binary classification metrics at standard thermostability threshold (>= 60°C)
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
    
    # Global AUC: mean ROC AUC across thresholds 30-90°C
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

def evaluate_mode(mode, device, base_dir, x_esm2, y_true, results_dir=None):
    if results_dir is None:
        results_dir = os.path.join(base_dir, 'results')
    use_ogt_feature = (mode in ['C', 'C2'])
    
    ensemble_preds = []
    
    for seed in CONFIG['seeds']:
        seed_dir = os.path.join(results_dir, f"seed{seed}")
        model_path = os.path.join(seed_dir, f'model_{mode}.pt')
        stage1_model_path = os.path.join(seed_dir, 'model_stage1.pt')
        
        if not os.path.exists(model_path):
            continue
            
        model = StableProtV7(
            emb_dim=CONFIG['input_size'],
            hidden=CONFIG['hidden_size'],
            bottleneck=CONFIG['bottleneck_size'],
            use_ogt_feature=use_ogt_feature
        ).to(device)
        sd = torch.load(model_path, map_location=device)
        load_compat_state_dict(model, sd)
        model.eval()
        
        # Predict OGT if needed
        ogt_pred = None
        if use_ogt_feature:
            stage1_predictor = StableProtV7(
                emb_dim=CONFIG['input_size'],
                hidden=CONFIG['hidden_size'],
                bottleneck=CONFIG['bottleneck_size']
            ).to(device)
            stage1_sd = torch.load(stage1_model_path, map_location=device)
            load_compat_state_dict(stage1_predictor, stage1_sd)
            stage1_predictor.eval()
            with torch.no_grad():
                ogt_pred = stage1_predictor(x_esm2, stage='ogt')
        
        with torch.no_grad():
            pred = model(x_esm2, stage='tm', ogt_pred=ogt_pred).cpu().numpy()
            
        ensemble_preds.append(pred)
        
    if not ensemble_preds:
        print(f"No trained models found for Mode {mode}")
        return None
        
    mean_preds = np.mean(ensemble_preds, axis=0)
    metrics = compute_metrics(y_true, mean_preds)
    
    # Calculate bootstrapped 95% CI for MAE and PCC (1000 resamples)
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default="results")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = os.path.dirname(os.path.abspath(__file__))
    holdout_path = os.path.join(base_dir, "../../../../data/test_data/fireprot_holdout_prott5.pt")
    
    if not os.path.exists(holdout_path):
        print(f"CRITICAL ERROR: Holdout dataset missing at {holdout_path}")
        return
        
    print("Loading FireProt holdout dataset...")
    d_fireprot = torch.load(holdout_path, weights_only=False)
    x_esm2 = d_fireprot['embeddings_esm2'].to(device)
    y_true = d_fireprot['temperatures'].numpy() if hasattr(d_fireprot['temperatures'], 'numpy') else np.array(d_fireprot['temperatures'])
    
    print(f"FireProt Holdout: {len(y_true)} testing targets.")
    
    # Resolve relative or absolute path for results_dir
    if not os.path.isabs(args.results_dir):
        resolved_results_dir = os.path.join(base_dir, args.results_dir)
    else:
        resolved_results_dir = args.results_dir
        
    modes = ['B', 'C', 'C2']
    results = {}
    
    print("\nEvaluating V7 Model Variants...")
    for mode in modes:
        res = evaluate_mode(mode, device, base_dir, x_esm2, y_true, results_dir=resolved_results_dir)
        if res is not None:
            preds, metrics = res
            results[mode] = {'preds': preds, 'metrics': metrics}
            
    # Print results table
    print("\n" + "="*120)
    print(f"{'V7 Model Variant':<20} | {'MAE (°C)':<15} | {'MAE 95% CI':<18} | {'PCC':<8} | {'PCC 95% CI':<18} | {'R²':<8} | {'Global AUC':<10}")
    print("="*120)
    
    for mode, data in results.items():
        m = data['metrics']
        mae_str = f"{m['mae']:.2f}"
        mae_ci_str = f"[{m['mae_ci'][0]:.2f}, {m['mae_ci'][1]:.2f}]"
        pcc_str = f"{m['pcc']:.2f}"
        pcc_ci_str = f"[{m['pcc_ci'][0]:.2f}, {m['pcc_ci'][1]:.2f}]"
        r2_str = f"{m['r2']:.2f}"
        auc_str = f"{m['global_auc']:.3f}"
        
        mode_name = f"V7 Mode {mode}"
        if mode == 'B':
            mode_name += " (Transfer)"
        elif mode == 'C':
            mode_name += " (Transfer+OGT)"
        elif mode == 'C2':
            mode_name += " (OGT only)"
            
        print(f"{mode_name:<20} | {mae_str:<15} | {mae_ci_str:<18} | {pcc_str:<8} | {pcc_ci_str:<18} | {r2_str:<8} | {auc_str:<10}")
    print("="*120)

    # Save results to json
    suffix = os.path.basename(args.results_dir.rstrip("/"))
    out_path = os.path.join(base_dir, f'v7_fireprot_eval_metrics_{suffix}.json')
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
