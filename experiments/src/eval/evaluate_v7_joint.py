import os
import sys
import torch
import numpy as np
import pandas as pd
from Bio import SeqIO
import subprocess

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.append(PROJECT_ROOT)

from experiments.src.training.v7_transfer.model import StableProtV7
from experiments.src.training.v7_transfer.config import CONFIG

def compute_metrics(y_true, y_pred, threshold=60.0):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    # Pearson
    pcc = np.corrcoef(y_true, y_pred)[0, 1] if len(np.unique(y_pred)) > 1 else 0.0
    
    # Spearman
    import scipy.stats
    spearman = scipy.stats.spearmanr(y_true, y_pred)[0] if len(np.unique(y_pred)) > 1 else 0.0
    
    # R2
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Classification at 60C
    y_true_b = (y_true >= threshold).astype(int)
    y_pred_b = (y_pred >= threshold).astype(int)
    
    from sklearn.metrics import confusion_matrix, f1_score
    cm = confusion_matrix(y_true_b, y_pred_b)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        acc = (tp + tn) / (tp + tn + fp + fn)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        from sklearn.metrics import matthews_corrcoef
        mcc = matthews_corrcoef(y_true_b, y_pred_b)
    else:
        acc = sens = spec = prec = mcc = 0.0
        
    f1 = f1_score(y_true_b, y_pred_b, zero_division=0)
    
    # MAPE
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1.0))) * 100
    
    # Top 10% enrichment
    n_top = max(1, int(0.1 * len(y_true)))
    top_pred_idx = np.argsort(y_pred)[-n_top:]
    top_true_idx = np.argsort(y_true)[-n_top:]
    enrich = len(set(top_pred_idx).intersection(set(top_true_idx))) / n_top
    
    try:
        from sklearn.metrics import roc_auc_score
        roc_auc = roc_auc_score(y_true_b, y_pred)
    except Exception:
        roc_auc = 0.5
        
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
        'mape': mape, 'enrich': enrich, 'roc_auc': roc_auc, 'global_auc': global_auc
    }

def load_compat_state_dict(model, state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            parts = k.split('.')
            layer_idx = int(parts[1])
            param_name = '.'.join(parts[2:])
            
            if layer_idx == 0:
                new_key = f"input_layer.{param_name}"
            elif layer_idx == 1:
                new_key = f"backbone_rest.0.{param_name}"
            elif layer_idx == 4:
                new_key = f"backbone_rest.3.{param_name}"
            elif layer_idx == 5:
                new_key = f"backbone_rest.4.{param_name}"
            elif layer_idx == 6:
                new_key = f"backbone_rest.5.{param_name}"
            else:
                new_key = k
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)

def evaluate_model(model_type, device):
    seeds = [1, 2, 3]
    models = []
    
    if model_type == 'esm2':
        results_dir = os.path.join(PROJECT_ROOT, "experiments/src/training/v7_transfer/results")
        emb_dim = 2560
        key_name = 'V7 ESM-2 Joint'
    else:
        results_dir = os.path.join(PROJECT_ROOT, "experiments/src/training/v7_transfer/results/saprot")
        emb_dim = 1280
        key_name = 'V7 SaProt Joint'
        
    for s in seeds:
        p = os.path.join(results_dir, f"seed{s}/model_joint.pt")
        if os.path.exists(p):
            model = StableProtV7(
                emb_dim=emb_dim,
                use_ogt_feature=False,
                use_tm_feature=False,
                hidden=CONFIG['hidden_size'],
                bottleneck=CONFIG['bottleneck_size']
            ).to(device)
            load_compat_state_dict(model, torch.load(p, map_location=device))
            model.eval()
            models.append(model)
            print(f"Loaded seed {s} joint {model_type.upper()} model.")
            
    if not models:
        print(f"No trained V7 Joint {model_type.upper()} models found.")
        return
        
    # --- ProThermDB Evaluation ---
    if model_type == 'esm2':
        print(f"\n--- ProThermDB {model_type.upper()} Evaluation ---")
        protherm_csv = os.path.join(PROJECT_ROOT, 'new_data/prothermdb_validation.csv')
        protherm_fasta = os.path.join(PROJECT_ROOT, 'new_data/prothermdb_validation.fasta')
        esm2_data_path = os.path.join(PROJECT_ROOT, "new_data/prepared_data_v2.pt")
        
        df_p = pd.read_csv(protherm_csv)
        protherm_dict = {str(row['UniProt_ID']): float(row['Tm']) for _, row in df_p.iterrows() if not np.isnan(row['Tm'])}
        
        protherm_seqs = []
        y_true_protherm_list = []
        for record in SeqIO.parse(protherm_fasta, 'fasta'):
            seq = str(record.seq)
            uid = record.id.split('|')[0]
            if uid in protherm_dict:
                protherm_seqs.append(seq)
                y_true_protherm_list.append(protherm_dict[uid])
        y_true_protherm = np.array(y_true_protherm_list)
        
        # Load and map ESM-2 embeddings
        d_esm2 = torch.load(esm2_data_path, map_location='cpu', weights_only=False)
        seq_to_esm2 = {str(seq).upper(): emb for seq, emb in zip(d_esm2['test_tm']['sequences'], d_esm2['test_tm']['embeddings'])}
        
        x_esm2_protherm_list = []
        for seq in protherm_seqs:
            x_esm2_protherm_list.append(seq_to_esm2[seq.upper()])
        x_esm2_protherm = torch.stack(x_esm2_protherm_list).to(device)
        
        # Inference with ensemble
        preds_list = []
        with torch.no_grad():
            for model in models:
                preds_list.append(model(x_esm2_protherm, stage='tm').cpu().numpy())
        y_pred_protherm = np.mean(preds_list, axis=0)
        
        m_protherm = compute_metrics(y_true_protherm, y_pred_protherm)
        print(f"{key_name} | ProTherm MAE: {m_protherm['mae']:.2f} | PCC: {m_protherm['pcc']:.2f} | R²: {m_protherm['r2']:.2f}")
        
        # Update protherm evaluation results pt file
        results_path = os.path.join(PROJECT_ROOT, "new_data/protherm_evaluation_results.pt")
        if os.path.exists(results_path):
            res = torch.load(results_path, map_location='cpu', weights_only=False)
            res['predictions'][key_name] = y_pred_protherm
            res['metrics'][key_name] = m_protherm
            torch.save(res, results_path)
            print(f"Updated ProThermDB predictions cache in {results_path}")
            
    # --- FireProtDB Evaluation ---
    print(f"\n--- FireProtDB {model_type.upper()} Evaluation ---")
    prott5_data_path = os.path.join(PROJECT_ROOT, "experiments/src/data/fireprot_holdout_prott5.pt")
    d_prott5 = torch.load(prott5_data_path, map_location='cpu', weights_only=False)
    y_true_fireprot = d_prott5['temperatures'].numpy() if hasattr(d_prott5['temperatures'], 'numpy') else np.array(d_prott5['temperatures'])
    
    if model_type == 'esm2':
        x_fireprot = d_prott5['embeddings_esm2'].to(device)
    else:
        saprot_data_path = os.path.join(PROJECT_ROOT, "data/test_data/fireprot_holdout_saprot.pt")
        d_saprot = torch.load(saprot_data_path, map_location='cpu', weights_only=False)
        x_fireprot = d_saprot['embeddings_saprot'].to(device)
        
    preds_list_fp = []
    with torch.no_grad():
        for model in models:
            preds_list_fp.append(model(x_fireprot, stage='tm').cpu().numpy())
    y_pred_fireprot = np.mean(preds_list_fp, axis=0)
    
    m_fireprot = compute_metrics(y_true_fireprot, y_pred_fireprot)
    print(f"{key_name} | FireProt MAE: {m_fireprot['mae']:.2f} | PCC: {m_fireprot['pcc']:.2f} | R²: {m_fireprot['r2']:.2f}")
    
    # Update fireprot evaluation results pt file
    results_path_fp = os.path.join(PROJECT_ROOT, "new_data/fireprot_evaluation_results.pt")
    if os.path.exists(results_path_fp):
        res_fp = torch.load(results_path_fp, map_location='cpu', weights_only=False)
        res_fp['predictions'][key_name] = y_pred_fireprot
        res_fp['metrics'][key_name] = m_fireprot
        torch.save(res_fp, results_path_fp)
        print(f"Updated FireProtDB predictions cache in {results_path_fp}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    evaluate_model('esm2', device)
    evaluate_model('saprot', device)
    
    # Rerun evaluate_temp_wise.py to update binned tables/plots
    print("\n--- Regenerating Temperature-Wise Benchmarks ---")
    cmd = [sys.executable, os.path.join(PROJECT_ROOT, "experiments/src/eval/evaluate_temp_wise.py")]
    subprocess.run(cmd, check=True)
    print("Temperature-wise benchmarks updated.")

if __name__ == "__main__":
    main()
