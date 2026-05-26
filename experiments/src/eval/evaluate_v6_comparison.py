import os
import sys
import torch
import numpy as np
import pandas as pd
import scipy.stats
from Bio import SeqIO

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

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
    
    k = max(1, int(0.1 * len(y_true)))
    t_top10 = np.percentile(y_true, 90)
    top_k_indices = np.argsort(y_pred)[-k:]
    enrich = np.sum(y_true[top_k_indices] >= t_top10) / float(k)
    
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

def main():
    print("="*60)
    print("V6 SEQUENCE-ALIGNED EMBEDDINGS COMPARISON (1894 Sequences)")
    print("="*60)
    
    # 1. Load ground truth validation sequences
    protherm_csv = os.path.join(PROJECT_ROOT, 'new_data/prothermdb_validation.csv')
    protherm_fasta = os.path.join(PROJECT_ROOT, 'new_data/prothermdb_validation.fasta')
    
    df_p = pd.read_csv(protherm_csv)
    protherm_dict = {str(row['UniProt_ID']): float(row['Tm']) for _, row in df_p.iterrows() if not pd.isna(row['Tm'])}
    
    protherm_seqs = []
    y_true_list = []
    for record in SeqIO.parse(protherm_fasta, 'fasta'):
        uid = record.id.split('|')[0]
        if uid in protherm_dict:
            protherm_seqs.append(str(record.seq).upper())
            y_true_list.append(protherm_dict[uid])
            
    # ProtT5 predictions align 1-to-1 with protherm_seqs
    prott5_preds_path = os.path.join(PROJECT_ROOT, "experiments/src/training/v5_multihead/results/ensemble/predictions.pt")
    if not os.path.exists(prott5_preds_path):
        print(f"Missing ProtT5 predictions at: {prott5_preds_path}")
        return
    prott5_data = torch.load(prott5_preds_path, map_location='cpu', weights_only=False)
    prott5_preds = prott5_data['y_pred'].numpy() if hasattr(prott5_data['y_pred'], 'numpy') else np.array(prott5_data['y_pred'])
    
    # 2. Load ESM-2 sequences and map predictions
    esm2_data_path = os.path.join(PROJECT_ROOT, "data/embeddings/prepared_data_v2.pt")
    esm2_preds_path = os.path.join(PROJECT_ROOT, "experiments/src/training/v6_multihead_esm2/results/ensemble/predictions.pt")
    if not os.path.exists(esm2_data_path) or not os.path.exists(esm2_preds_path):
        print("Missing ESM-2 dataset or predictions.")
        return
    esm2_data = torch.load(esm2_data_path, map_location='cpu', weights_only=False)
    esm2_preds_data = torch.load(esm2_preds_path, map_location='cpu', weights_only=False)
    esm2_preds_all = esm2_preds_data['y_pred'].numpy() if hasattr(esm2_preds_data['y_pred'], 'numpy') else np.array(esm2_preds_data['y_pred'])
    esm2_seq_to_idx = {str(seq).upper(): idx for idx, seq in enumerate(esm2_data['test_tm']['sequences'])}
    
    # 3. Load SaProt sequences and map predictions
    saprot_data_path = os.path.join(PROJECT_ROOT, "data/embeddings/prepared_data_v4_saprot.pt")
    saprot_preds_path = os.path.join(PROJECT_ROOT, "experiments/src/training/v6_multihead_saprot/results/ensemble/predictions.pt")
    if not os.path.exists(saprot_data_path) or not os.path.exists(saprot_preds_path):
        print("Missing SaProt dataset or predictions.")
        return
    saprot_data = torch.load(saprot_data_path, map_location='cpu', weights_only=False)
    saprot_preds_data = torch.load(saprot_preds_path, map_location='cpu', weights_only=False)
    saprot_preds_all = saprot_preds_data['y_pred'].numpy() if hasattr(saprot_preds_data['y_pred'], 'numpy') else np.array(saprot_preds_data['y_pred'])
    saprot_seq_to_idx = {str(seq).upper(): idx for idx, seq in enumerate(saprot_data['test_tm']['sequences'])}
    
    # Align to intersection
    aligned_indices = []
    for idx, seq in enumerate(protherm_seqs):
        if seq in esm2_seq_to_idx and seq in saprot_seq_to_idx:
            aligned_indices.append((idx, esm2_seq_to_idx[seq], saprot_seq_to_idx[seq]))
            
    print(f"Successfully aligned {len(aligned_indices)} sequences across ProtT5, ESM-2, and SaProt.")
    
    y_true_aligned = []
    prott5_aligned = []
    esm2_aligned = []
    saprot_aligned = []
    
    for idx_pt, idx_esm, idx_sap in aligned_indices:
        y_true_aligned.append(y_true_list[idx_pt])
        prott5_aligned.append(prott5_preds[idx_pt])
        esm2_aligned.append(esm2_preds_all[idx_esm])
        saprot_aligned.append(saprot_preds_all[idx_sap])
        
    y_true_aligned = np.array(y_true_aligned)
    prott5_aligned = np.array(prott5_aligned)
    esm2_aligned = np.array(esm2_aligned)
    saprot_aligned = np.array(saprot_aligned)
    
    # Compute metrics
    m_prott5 = compute_metrics(y_true_aligned, prott5_aligned)
    m_esm2 = compute_metrics(y_true_aligned, esm2_aligned)
    m_saprot = compute_metrics(y_true_aligned, saprot_aligned)
    
    results = {
        'ProtT5': m_prott5,
        'ESM-2': m_esm2,
        'SaProt': m_saprot
    }
    
    df = pd.DataFrame(results).T
    df = df[['mae', 'rmse', 'pcc', 'spearman', 'r2', 'roc_auc', 'global_auc']]
    df.columns = ['MAE (°C)', 'RMSE (°C)', 'Pearson (PCC)', 'Spearman', 'R²', 'ROC AUC (60°C)', 'Global AUC']
    
    print("\nComparison Table (Aligned 1894 Subset):")
    headers = ["Model"] + list(df.columns)
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for idx, row in df.iterrows():
        cols = [str(idx)] + [f"{val:.4f}" for val in row.values]
        print(" | ".join(cols))
        
    # Save table
    table_dir = os.path.join(PROJECT_ROOT, "paper/writeup/tables")
    os.makedirs(table_dir, exist_ok=True)
    with open(os.path.join(table_dir, "v6_embeddings_comparison.md"), 'w') as f:
        f.write(" | ".join(headers) + "\n")
        f.write(" | ".join(["---"] * len(headers)) + "\n")
        for idx, row in df.iterrows():
            cols = [str(idx)] + [f"{val:.4f}" for val in row.values]
            f.write(" | ".join(cols) + "\n")
    print(f"\nSaved aligned comparison table to {os.path.join(table_dir, 'v6_embeddings_comparison.md')}")

if __name__ == "__main__":
    main()
