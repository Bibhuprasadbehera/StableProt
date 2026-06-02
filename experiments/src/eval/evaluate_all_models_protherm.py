import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats
import pandas as pd
from Bio import SeqIO

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.append(PROJECT_ROOT)

# -------------------------------------------------------------
# Model Definitions
# -------------------------------------------------------------
class MLP_C2H2(nn.Module):
    def __init__(self, input_size=1024, hidden_size_1=512, hidden_size_2=256):
        super().__init__()
        self.model = nn.ModuleList([
            nn.Linear(input_size, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, 1),
            nn.Sigmoid()
        ])
    def forward(self, point):
        for layer in self.model:
            point = layer(point)
        return point

class MLP_Baseline(nn.Module):
    def __init__(self, input_size=1024, hidden_size_1=256, hidden_size_2=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

class MLP_Improved(nn.Module):
    def __init__(self, input_size=1024, hidden_size_1=512, hidden_size_2=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.BatchNorm1d(hidden_size_1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.BatchNorm1d(hidden_size_2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size_2, 1)
        )
    def forward(self, x):
        return self.model(x)

class MLP_Regression(nn.Module):
    def __init__(self, input_size=1024, hidden_size_1=512, hidden_size_2=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.BatchNorm1d(hidden_size_1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.BatchNorm1d(hidden_size_2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size_2, 1)
        )
    def forward(self, x):
        return self.model(x)

class MLP_Regression_Improved(nn.Module):
    def __init__(self, input_size=1024, hidden_size_1=512, hidden_size_2=256):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.bn1 = nn.BatchNorm1d(hidden_size_1)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.bn2 = nn.BatchNorm1d(hidden_size_2)
        self.dropout2 = nn.Dropout(0.2)
        self.residual_proj = nn.Linear(hidden_size_1, hidden_size_2) if hidden_size_1 != hidden_size_2 else nn.Identity()
        self.head = nn.Linear(hidden_size_2, 1)
    def forward(self, x):
        x1 = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x2 = self.dropout2(F.relu(self.bn2(self.fc2(x1)) + self.residual_proj(x1)))
        return self.head(x2)

class MultiHead_TmPredictor(nn.Module):
    def __init__(self, input_size=2560, hidden1=512, hidden2=256, dropout1=0.3, dropout2=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.dropout1 = nn.Dropout(dropout1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.dropout2 = nn.Dropout(dropout2)
        self.residual_proj = nn.Linear(hidden1, hidden2) if hidden1 != hidden2 else nn.Identity()
        self.head_ogt = nn.Linear(hidden2, 1)
        self.head_tm = nn.Linear(hidden2, 1)
    def forward(self, x, head='tm'):
        x1 = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x2 = self.dropout2(F.relu(self.bn2(self.fc2(x1)) + self.residual_proj(x1)))
        if head == 'ogt':
            return self.head_ogt(x2).squeeze(-1)
        else:
            return self.head_tm(x2).squeeze(-1)



# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def load_v0_model(model_path, device='cpu'):
    state_dict = torch.load(model_path, map_location=torch.device(device), weights_only=False)['state_dict']
    new_state_dict = {}
    for key in list(state_dict.keys()):
        new_key = key.replace('model.model.', 'model.')
        new_state_dict[new_key] = state_dict[key]
    h1 = new_state_dict['model.0.weight'].shape[0]
    h2 = new_state_dict['model.2.weight'].shape[0]
    input_size = new_state_dict['model.0.weight'].shape[1]
    classifier = MLP_C2H2(input_size=input_size, hidden_size_1=h1, hidden_size_2=h2)
    classifier.load_state_dict(new_state_dict)
    classifier.eval()
    classifier.to(device)
    return classifier

def compute_expected_temperatures(prob_matrix, thresholds):
    step_sizes = np.diff(thresholds)
    step = step_sizes[0] if len(step_sizes) > 0 else 5
    base_temp = max(0, thresholds[0] - step)
    y_pred = np.full(prob_matrix.shape[0], base_temp, dtype=float)
    for i in range(len(thresholds)):
        y_pred += prob_matrix[:, i] * step
    return y_pred

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing ProThermDB evaluations on device: {device}")
    
    # -------------------------------------------------------------
    # Load ProThermDB Datasets
    # -------------------------------------------------------------
    protherm_csv = os.path.join(PROJECT_ROOT, 'new_data/prothermdb_validation.csv')
    protherm_fasta = os.path.join(PROJECT_ROOT, 'new_data/prothermdb_validation.fasta')
    prott5_data_path = os.path.join(PROJECT_ROOT, "new_data/prepared_data_v5_prott5.pt")
    esm2_data_path = os.path.join(PROJECT_ROOT, "new_data/prepared_data_v2.pt")
    
    df_p = pd.read_csv(protherm_csv)
    protherm_dict = {str(row['UniProt_ID']): float(row['Tm']) for _, row in df_p.iterrows() if not np.isnan(row['Tm'])}
    
    protherm_seqs = []
    y_true_list = []
    for record in SeqIO.parse(protherm_fasta, 'fasta'):
        seq = str(record.seq)
        uid = record.id.split('|')[0]
        if uid in protherm_dict:
            protherm_seqs.append(seq)
            y_true_list.append(protherm_dict[uid])
            
    y_true = np.array(y_true_list)
    print(f"Loaded {len(protherm_seqs)} ProThermDB validation sequences and ground truths.")
    
    # Load ProtT5 embeddings
    d_prott5 = torch.load(prott5_data_path, map_location='cpu', weights_only=False)
    x_prott5 = d_prott5['test_tm']['embeddings'].to(device)
    
    # Load and map ESM-2 embeddings
    print("Loading and mapping ESM-2 embeddings...")
    d_esm2 = torch.load(esm2_data_path, map_location='cpu', weights_only=False)
    seq_to_esm2 = {str(seq).upper(): emb for seq, emb in zip(d_esm2['test_tm']['sequences'], d_esm2['test_tm']['embeddings'])}
    
    x_esm2_list = []
    for seq in protherm_seqs:
        x_esm2_list.append(seq_to_esm2[seq.upper()])
    x_esm2 = torch.stack(x_esm2_list).to(device)
    print(f"ESM-2 embeddings loaded. Shape: {x_esm2.shape}")
    
    results = {}
    
    # ── V0 Original ──
    print("\nEvaluating V0 Original...")
    v0_thresholds = [40, 45, 50, 55, 60, 65]
    v0_models_dir = os.path.join(PROJECT_ROOT, "benchmark_models/StableProt/models")
    v0_probs = []
    for t in v0_thresholds:
        t_probs = []
        for s in range(1, 6):
            p = os.path.join(v0_models_dir, f"mean_major_imbal-{t}_s{s}.pt")
            if os.path.exists(p):
                model = load_v0_model(p, device=device)
                with torch.no_grad():
                    out = model(x_prott5.float()).squeeze().cpu().numpy()
                t_probs.append(out)
        if t_probs:
            v0_probs.append(np.mean(t_probs, axis=0))
    if v0_probs:
        v0_preds = compute_expected_temperatures(np.column_stack(v0_probs), v0_thresholds)
        results['V0 Original'] = {'y_pred': v0_preds, 'type': 'Binary Proxy'}

    # ── V1 Baseline ──
    print("Evaluating V1 Baseline...")
    v1_thresholds = list(range(5, 100, 5))
    v1_probs = []
    for t in v1_thresholds:
        t_probs = []
        for s in range(1, 6):
            p = os.path.join(PROJECT_ROOT, f"experiments/src/training/v1_baseline/results/t{t}/seed{s}/model.pt")
            if os.path.exists(p):
                model = MLP_Baseline().to(device)
                model.load_state_dict(torch.load(p, map_location=device, weights_only=False))
                model.eval()
                with torch.no_grad():
                    out = model(x_prott5.float()).squeeze().cpu().numpy()
                t_probs.append(out)
        if t_probs:
            v1_probs.append(np.mean(t_probs, axis=0))
    if v1_probs:
        v1_preds = compute_expected_temperatures(np.column_stack(v1_probs), v1_thresholds)
        results['V1 Baseline'] = {'y_pred': v1_preds, 'type': 'Binary Proxy'}

    # ── V2 Improved ──
    print("Evaluating V2 Improved...")
    v2_thresholds = list(range(5, 100, 5))
    v2_probs = []
    for t in v2_thresholds:
        t_probs = []
        for s in range(1, 6):
            p = os.path.join(PROJECT_ROOT, f"experiments/src/training/v2_improved/results/t{t}/seed{s}/model.pt")
            if os.path.exists(p):
                model = MLP_Improved().to(device)
                model.load_state_dict(torch.load(p, map_location=device, weights_only=False))
                model.eval()
                with torch.no_grad():
                    logits = model(x_prott5.float()).squeeze()
                    out = torch.sigmoid(logits).cpu().numpy()
                t_probs.append(out)
        if t_probs:
            v2_probs.append(np.mean(t_probs, axis=0))
    if v2_probs:
        v2_preds = compute_expected_temperatures(np.column_stack(v2_probs), v2_thresholds)
        results['V2 Improved'] = {'y_pred': v2_preds, 'type': 'Binary Proxy'}

    # ── V3 Regression ──
    print("Evaluating V3 Regression...")
    v3_preds = []
    for s in range(1, 6):
        p = os.path.join(PROJECT_ROOT, f"experiments/src/training/v3_regression/results/seed{s}/model.pt")
        if os.path.exists(p):
            model = MLP_Regression().to(device)
            model.load_state_dict(torch.load(p, map_location=device, weights_only=False))
            model.eval()
            with torch.no_grad():
                out = model(x_prott5.float()).squeeze().cpu().numpy()
            v3_preds.append(out)
    if v3_preds:
        results['V3 Regression'] = {'y_pred': np.mean(v3_preds, axis=0), 'type': 'Continuous Proxy'}

    # ── V4 Improved Regression ──
    print("Evaluating V4 Improved Regression...")
    v4_preds = []
    for s in range(1, 6):
        p = os.path.join(PROJECT_ROOT, f"experiments/src/training/v4_improved/results/seed{s}/model.pt")
        if os.path.exists(p):
            model = MLP_Regression_Improved().to(device)
            model.load_state_dict(torch.load(p, map_location=device, weights_only=False))
            model.eval()
            with torch.no_grad():
                out = model(x_prott5.float()).squeeze().cpu().numpy()
            v4_preds.append(out)
    if v4_preds:
        results['V4 Improved Regr.'] = {'y_pred': np.mean(v4_preds, axis=0), 'type': 'Continuous Proxy'}

    # ── V5 Multi-Head (ProtT5) ──
    print("Evaluating V5 Multi-Head (ProtT5)...")
    v5_preds = []
    for s in range(1, 6):
        p = os.path.join(PROJECT_ROOT, f"experiments/src/training/v5_multihead/results/seed{s}/model.pt")
        if os.path.exists(p):
            model = MultiHead_TmPredictor(input_size=1024).to(device)
            model.load_state_dict(torch.load(p, map_location=device, weights_only=False))
            model.eval()
            with torch.no_grad():
                out = model(x_prott5.float(), head='tm').squeeze().cpu().numpy()
            v5_preds.append(out)
    if v5_preds:
        results['V5 Multi-Head (ProtT5)'] = {'y_pred': np.mean(v5_preds, axis=0), 'type': 'Dedicated Tm Head'}

    # ── V6 SaProt (Weighted Multi-Head) ──
    print("Evaluating V6 SaProt...")
    # Load SaProt embeddings for sequence matching
    saprot_data_path = os.path.join(PROJECT_ROOT, "data/embeddings/prepared_data_v4_saprot_cleaned.pt")
    if os.path.exists(saprot_data_path):
        saprot_data = torch.load(saprot_data_path, map_location='cpu', weights_only=False)
        seq_to_saprot = {}
        for split in ['train_tm', 'val_tm', 'test_tm']:
            if split in saprot_data and 'sequences' in saprot_data[split]:
                for seq, emb in zip(saprot_data[split]['sequences'], saprot_data[split]['embeddings']):
                    seq_to_saprot[str(seq).upper()] = emb
        
        # Match ProThermDB sequences to SaProt embeddings
        saprot_indices = []
        saprot_embs = []
        for idx, seq in enumerate(protherm_seqs):
            if seq.upper() in seq_to_saprot:
                saprot_indices.append(idx)
                saprot_embs.append(seq_to_saprot[seq.upper()])
        
        if saprot_embs:
            x_saprot = torch.stack(saprot_embs).to(device)
            v6_saprot_preds = []
            for s in range(1, 6):
                p = os.path.join(PROJECT_ROOT, f"experiments/src/training/v6_saprot/results/seed{s}/model.pt")
                if os.path.exists(p):
                    model = MultiHead_TmPredictor(input_size=1280).to(device)
                    # V6 SaProt uses separate input layers (MultiHead_SaProtPredictor architecture)
                    # but shares the same state_dict key naming convention
                    try:
                        model.load_state_dict(torch.load(p, map_location=device, weights_only=False))
                    except RuntimeError:
                        # Architecture mismatch — use SaProtPredictor class from fireprot eval
                        from evaluate_all_models_fireprot import MultiHead_SaProtPredictor
                        model = MultiHead_SaProtPredictor().to(device)
                        model.load_state_dict(torch.load(p, map_location=device, weights_only=False))
                    model.eval()
                    with torch.no_grad():
                        out = model(x_saprot.float(), head='tm').squeeze().cpu().numpy()
                    v6_saprot_preds.append(out)
            
            if v6_saprot_preds:
                ensemble_saprot = np.mean(v6_saprot_preds, axis=0)
                # Build full prediction array: SaProt where available, ESM-2 V5 fallback
                v6_full = np.copy(results.get('V5 Multi-Head (ProtT5)', {}).get('y_pred', np.zeros(len(y_true))))
                for i, idx in enumerate(saprot_indices):
                    v6_full[idx] = ensemble_saprot[i]
                results['V6 SaProt'] = {'y_pred': v6_full, 'type': 'Dedicated Tm Head'}
                print(f"  Matched {len(saprot_indices)}/{len(protherm_seqs)} with SaProt, rest fallback")
    else:
        print("WARNING: SaProt data not found.")

    # ── Load Baselines (TemBERTure, ESMStabP, DeepSTABp, ThermoFormer) ──
    baseline_path = os.path.join(PROJECT_ROOT, "new_data/baseline_predictions.pt")
    if os.path.exists(baseline_path):
        print("Loading baseline predictions...")
        baselines = torch.load(baseline_path, map_location='cpu', weights_only=False)
        results['TemBERTure'] = {'y_pred': baselines['protherm']['temberture'], 'type': 'Continuous Proxy'}
        results['ESMStabP'] = {'y_pred': baselines['protherm']['esmstabp'], 'type': 'Continuous Proxy'}
        if 'deepstabp' in baselines['protherm']:
            results['DeepSTABp'] = {'y_pred': baselines['protherm']['deepstabp'], 'type': 'Continuous Proxy'}
        if 'thermoformer' in baselines['protherm']:
            results['ThermoFormer'] = {'y_pred': baselines['protherm']['thermoformer'], 'type': 'Continuous Proxy'}
    else:
        print("WARNING: Baseline predictions file missing.")

    # -------------------------------------------------------------
    # Compute Metrics and Display Summary
    # -------------------------------------------------------------
    print("\n" + "=" * 175)
    print("  PROTHERMDB VALIDATION BENCHMARK (ALL MODELS V0 TO V6 + BASELINES)")
    print("" + "=" * 175)
    print(f"{'Model Iteration':<28} | {'Type':<18} | {'MAE':<6} | {'RMSE':<6} | {'PCC':<5} | {'Spearman':<8} | {'R²':<6} | {'MCC':<6} | {'F1':<5} | {'AUC':<5} | {'Global AUC':<10} | {'Top-10% Enrich':<14}")
    print("-" * 175)
    
    metrics_summary = {}
    for name, data in results.items():
        m = compute_metrics(y_true, data['y_pred'])
        metrics_summary[name] = m
        print(f"{name:<28} | {data['type']:<18} | {m['mae']:<6.2f} | {m['rmse']:<6.2f} | {m['pcc']:<5.2f} | {m['spearman']:<8.2f} | {m['r2']:<6.2f} | {m['mcc']:<6.3f} | {m['f1']:<5.2f} | {m['roc_auc']:<5.2f} | {m['global_auc']:<10.3f} | {m['enrich']:<14.3f}")
    print("-" * 175)

    # Save all results to a single pt file for final plotting and papers
    out_results_path = os.path.join(PROJECT_ROOT, "new_data/protherm_evaluation_results.pt")
    save_dict = {
        'y_true': y_true,
        'predictions': {name: data['y_pred'] for name, data in results.items()},
        'metrics': metrics_summary
    }
    torch.save(save_dict, out_results_path)
    print(f"Saved evaluation results and metrics to {out_results_path}")

if __name__ == "__main__":
    main()
