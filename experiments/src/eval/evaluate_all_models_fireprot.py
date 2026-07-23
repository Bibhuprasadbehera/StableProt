import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats
import pandas as pd

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

class MultiHead_SaProtPredictor(nn.Module):
    def __init__(self, hidden1=512, hidden2=256, dropout1=0.3, dropout2=0.2):
        super().__init__()
        # TM head layers
        self.input_layer_tm = nn.Linear(1280, hidden1)
        self.bn1_tm = nn.BatchNorm1d(hidden1)
        self.fc2_tm = nn.Linear(hidden1, hidden2)
        self.bn2_tm = nn.BatchNorm1d(hidden2)
        self.residual_proj_tm = nn.Linear(hidden1, hidden2) if hidden1 != hidden2 else nn.Identity()
        self.head_tm = nn.Linear(hidden2, 1)
        
        # OGT head layers
        self.input_layer_ogt = nn.Linear(2560, hidden1)
        self.bn1_ogt = nn.BatchNorm1d(hidden1)
        self.fc2_ogt = nn.Linear(hidden1, hidden2)
        self.bn2_ogt = nn.BatchNorm1d(hidden2)
        self.residual_proj_ogt = nn.Linear(hidden1, hidden2) if hidden1 != hidden2 else nn.Identity()
        self.head_ogt = nn.Linear(hidden2, 1)
        
        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)
        
    def forward(self, x, head='tm'):
        if head == 'tm':
            x1 = self.input_layer_tm(x)
            x1 = self.dropout1(F.relu(self.bn1_tm(x1)))
            x2 = self.dropout2(F.relu(self.bn2_tm(self.fc2_tm(x1)) + self.residual_proj_tm(x1)))
            return self.head_tm(x2).squeeze(-1)
        else:
            x1 = self.input_layer_ogt(x)
            x1 = self.dropout1(F.relu(self.bn1_ogt(x1)))
            x2 = self.dropout2(F.relu(self.bn2_ogt(self.fc2_ogt(x1)) + self.residual_proj_ogt(x1)))
            return self.head_ogt(x2).squeeze(-1)



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

def compute_metrics(y_true, y_pred, y_conf=None):
    mae = np.mean(np.abs(y_true - y_pred))
    if y_conf is not None:
        int_err = np.maximum(0.0, np.abs(y_true - y_pred) - y_conf)
        interval_mae = np.mean(int_err)
    else:
        interval_mae = mae
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
        'mae': float(mae),
        'interval_mae': float(interval_mae),
        'rmse': float(rmse),
        'pcc': float(pcc),
        'spearman': float(spearman),
        'r2': float(r2),
        'acc': float(acc),
        'sens': float(sens),
        'spec': float(spec),
        'prec': float(prec),
        'f1': float(f1),
        'mcc': float(mcc),
        'mape': float(mape),
        'enrich': float(enrich),
        'roc_auc': float(roc_auc),
        'global_auc': float(global_auc)
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing FireProt evaluations on device: {device}")
    
    # -------------------------------------------------------------
    # Load FireProt Datasets
    # -------------------------------------------------------------
    prott5_data_path = os.path.join(PROJECT_ROOT, "experiments/src/data/fireprot_holdout_prott5.pt")
    saprot_data_path = os.path.join(PROJECT_ROOT, "data/test_data/fireprot_holdout_saprot.pt")
    
    d_prott5 = torch.load(prott5_data_path, map_location='cpu', weights_only=False)
    d_saprot = torch.load(saprot_data_path, map_location='cpu', weights_only=False)
    
    # Load V7 & V8 training sequences to eliminate evaluation overlap contamination
    v7_train_path = os.path.join(PROJECT_ROOT, "data/embeddings/prepared_data_v7_saprot1.3b_seqonly.pt")
    train_seqs_set = set()
    if os.path.exists(v7_train_path):
        v7_data_tmp = torch.load(v7_train_path, map_location='cpu', weights_only=False)
        if 'train_tm' in v7_data_tmp and 'sequences' in v7_data_tmp['train_tm']:
            train_seqs_set = {str(s).upper() for s in v7_data_tmp['train_tm']['sequences']}
    v8_train_path = os.path.join(PROJECT_ROOT, "data/embeddings/saprot_tm_struct_embeddings.pt")
    if os.path.exists(v8_train_path):
        v8_data_tmp = torch.load(v8_train_path, map_location='cpu', weights_only=False)
        if 'train_tm' in v8_data_tmp and 'sequences' in v8_data_tmp['train_tm']:
            train_seqs_set.update({str(s).upper() for s in v8_data_tmp['train_tm']['sequences']})

    seqs_all = [str(s) for s in d_saprot['sequences']]
    kept_indices = [i for i, s in enumerate(seqs_all) if s.upper() not in train_seqs_set]
    print(f"Loaded {len(kept_indices)}/{len(seqs_all)} FireProtDB validation sequences (decontaminated against V7/V8 train).")

    x_prott5 = d_prott5['embeddings_prott5'].to(device)[kept_indices]
    x_esm2 = d_prott5['embeddings_esm2'].to(device)[kept_indices]
    y_true_raw = d_prott5['temperatures'].numpy() if hasattr(d_prott5['temperatures'], 'numpy') else np.array(d_prott5['temperatures'])
    y_true = y_true_raw[kept_indices]
    x_saprot = d_saprot['embeddings_saprot'].to(device)[kept_indices]
    seqs_fp = [seqs_all[i] for i in kept_indices]
    
    results = {}
    
    # ── V0 Original ──
    print("\nEvaluating V0 Original (TemStaPro)...")
    v0_thresholds = [40, 45, 50, 55, 60, 65]
    v0_models_dir = os.path.join(PROJECT_ROOT, "benchmark_models_tm/StableProt/models")
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
        results['TemStaPro'] = {'y_pred': v0_preds, 'type': 'Binary Proxy'}

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
    v6_saprot_preds = []
    for s in range(1, 6):
        p = os.path.join(PROJECT_ROOT, f"experiments/src/training/v6_saprot/results/seed{s}/model.pt")
        if os.path.exists(p):
            model = MultiHead_SaProtPredictor().to(device)
            model.load_state_dict(torch.load(p, map_location=device, weights_only=False))
            model.eval()
            with torch.no_grad():
                out = model(x_saprot.float(), head='tm').squeeze().cpu().numpy()
            v6_saprot_preds.append(out)
    if v6_saprot_preds:
        results['StableProt'] = {'y_pred': np.mean(v6_saprot_preds, axis=0), 'type': 'Dedicated Tm Head'}

    # ── V7 SaProt (Shared Backbone) ──
    print("Evaluating V7 SaProt...")
    v7_saprot_preds = []
    
    # Import MultiHeadSaProtV7 dynamically
    sys.path.append(os.path.join(PROJECT_ROOT, "experiments/src/training/v7_shared"))
    from train import MultiHeadSaProtV7
    
    for s in range(1, 6):
        p = os.path.join(PROJECT_ROOT, f"experiments/src/training/v7_shared/results/seed{s}/best_model.pt")
        if os.path.exists(p):
            model = MultiHeadSaProtV7(input_dim=1280).to(device)
            model.load_state_dict(torch.load(p, map_location=device, weights_only=False))
            model.eval()
            with torch.no_grad():
                out = model(x_saprot.float(), task='tm').squeeze().cpu().numpy()
    if v7_saprot_preds:
        results['StableProt V7'] = {'y_pred': np.mean(v7_saprot_preds, axis=0), 'type': 'Dedicated Tm Head'}

    # ── V8/V9 SaProt (Disjoint Backbone with 2-stage inference) ──
    VERSION = os.environ.get("STABLEPROT_VERSION", "v8_disjoint")
    label_version = "StableProt V9"
    print(f"Evaluating {label_version}...")
    import importlib.util
    import inspect
    v8_dir = os.path.join(PROJECT_ROOT, f"experiments/src/training/{VERSION}")
    if v8_dir not in sys.path:
        sys.path.insert(0, v8_dir)
    v8_train_path = os.path.join(v8_dir, "train.py")
    spec = importlib.util.spec_from_file_location("train_v8", v8_train_path)
    train_v8 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_v8)
    MultiHeadSaProtV8 = train_v8.MultiHeadSaProtV8
    enrich_inputs_v8 = train_v8.enrich_inputs
    sanitize_data_v8 = train_v8.sanitize_data
    
    tm_data_v8 = torch.load(os.path.join(PROJECT_ROOT, "data/embeddings/saprot_tm_struct_embeddings.pt"), map_location='cpu', weights_only=False)
    _, _, tr_lbl_v8, _, _ = sanitize_data_v8(tm_data_v8['train_tm'], is_tm=True)
    tm_mean_v8, tm_std_v8 = tr_lbl_v8.mean().item(), tr_lbl_v8.std().item()

    seqs_fp = [seqs_all[i] for i in kept_indices]
    v8_mus = []
    v8_vars = []
    struct_path = os.path.join(PROJECT_ROOT, "data/embeddings/fireprot_v8_struct_embeddings.pt")
    if os.path.exists(struct_path):
        d_struct = torch.load(struct_path, map_location='cpu', weights_only=False)
        embs_v8_list = [d_struct.get(seq, x_saprot[i].cpu()) for i, seq in enumerate(seqs_fp)]
        embs_v8 = torch.stack(embs_v8_list, dim=0)
    else:
        embs_v8 = x_saprot.cpu()

    sig = inspect.signature(MultiHeadSaProtV8.__init__)
    model_kwargs = {}
    if 'use_residuals' in sig.parameters:
        model_kwargs['use_residuals'] = train_v8.CONFIG.get('use_residuals', True)

    for s in range(1, 6):
        pt_tm = os.path.join(PROJECT_ROOT, f"experiments/src/training/{VERSION}/results/seed{s}/model_tm.pt")
        pt_ogt = os.path.join(PROJECT_ROOT, f"experiments/src/training/{VERSION}/results/seed{s}/model_ogt.pt")
        pt_comb = os.path.join(PROJECT_ROOT, f"experiments/src/training/{VERSION}/results/seed{s}/model.pt")
        norm_p = os.path.join(PROJECT_ROOT, f"experiments/src/training/{VERSION}/results/seed{s}/normalization_stats.pt")
        if not os.path.exists(norm_p):
            norm_p = os.path.join(PROJECT_ROOT, f"experiments/src/training/{VERSION}/results/normalization_stats.pt")
        norms = torch.load(norm_p, map_location='cpu', weights_only=False) if os.path.exists(norm_p) else {}
        
        m_t, m_o = None, None
        if os.path.exists(pt_tm) and os.path.exists(pt_ogt):
            m_t = MultiHeadSaProtV8(**model_kwargs).to(device)
            m_t.load_state_dict(torch.load(pt_tm, map_location=device, weights_only=False))
            m_o = MultiHeadSaProtV8(**model_kwargs).to(device)
            m_o.load_state_dict(torch.load(pt_ogt, map_location=device, weights_only=False))
        elif os.path.exists(pt_comb):
            m_comb = MultiHeadSaProtV8(**model_kwargs).to(device)
            m_comb.load_state_dict(torch.load(pt_comb, map_location=device, weights_only=False))
            m_t, m_o = m_comb, m_comb
        if m_t is not None and m_o is not None:
            m_t.eval()
            m_o.eval()
            with torch.no_grad():
                emb_o, aux_o = enrich_inputs_v8(embs_v8, seqs_fp, tmhmm_flags=None, ogt_priors=None)
                pred_ogt = m_o(emb_o.to(device), aux_o.to(device), head='ogt').cpu()
                if 'ogt_mean' in norms and 'ogt_std' in norms:
                    pred_ogt = pred_ogt * norms['ogt_std'] + norms['ogt_mean']
                emb_t, aux_t = enrich_inputs_v8(embs_v8, seqs_fp, tmhmm_flags=None, ogt_priors=pred_ogt.numpy())
                z_mu, z_lv = m_t(emb_t.to(device), aux_t.to(device), head='tm')
                out_mu = (z_mu.cpu() * tm_std_v8 + tm_mean_v8).numpy()
                out_var = (z_lv.cpu() * (tm_std_v8 ** 2)).numpy()
            v8_mus.append(out_mu)
            v8_vars.append(out_var)

    if v8_mus:
        mus_stack = np.stack(v8_mus, axis=0)
        vars_stack = np.stack(v8_vars, axis=0)
        weights = 1.0 / (vars_stack + 1e-6)
        ens_mu = np.sum(mus_stack * weights, axis=0) / np.sum(weights, axis=0)
        ens_sigma = np.sqrt(1.0 / np.sum(weights, axis=0) + np.var(mus_stack, axis=0))
        results[label_version] = {
            "y_pred": ens_mu,
            "y_conf": ens_sigma,
            "type": "Dedicated Tm Head",
        }
        print(f"  {label_version} evaluated on FireProtDB sequences via confidence-weighted 2-stage inference")

    # ── Load Baselines (TemBERTure, ESMStabP, DeepSTABp, ThermoFormer) ──
    baseline_path = os.path.join(PROJECT_ROOT, "new_data/baseline_predictions.pt")
    if os.path.exists(baseline_path):
        print("Loading baseline predictions...")
        baselines = torch.load(baseline_path, map_location='cpu', weights_only=False)
        results['TemBERTure'] = {'y_pred': np.array(baselines['fireprot']['temberture'])[kept_indices], 'type': 'Continuous Proxy'}
        results['ESMStabP'] = {'y_pred': np.array(baselines['fireprot']['esmstabp'])[kept_indices], 'type': 'Continuous Proxy'}
        if 'deepstabp' in baselines['fireprot']:
            results['DeepSTABp'] = {'y_pred': np.array(baselines['fireprot']['deepstabp'])[kept_indices], 'type': 'Continuous Proxy'}
        if 'thermoformer' in baselines['fireprot']:
            results['ThermoFormer'] = {'y_pred': np.array(baselines['fireprot']['thermoformer'])[kept_indices], 'type': 'Continuous Proxy'}
    else:
        print("WARNING: Baseline predictions file missing.")

    # -------------------------------------------------------------
    # Compute Metrics and Display Summary
    # -------------------------------------------------------------
    print("\n" + "=" * 175)
    print("  FIREPROT HOLDOUT BENCHMARK (ALL MODELS V0 TO V6 + BASELINES)")
    print("" + "=" * 175)
    print(f"{'Model Iteration':<28} | {'Type':<18} | {'MAE':<6} | {'Int-MAE':<7} | {'RMSE':<6} | {'PCC':<5} | {'Spearman':<8} | {'R²':<6} | {'MCC':<6} | {'F1':<5} | {'AUC':<5} | {'Global AUC':<10} | {'Top-10% Enrich':<14}")
    print("-" * 185)
    
    metrics_summary = {}
    for name, data in results.items():
        m = compute_metrics(y_true, data['y_pred'], data.get('y_conf', None))
        metrics_summary[name] = m
        print(f"{name:<28} | {data['type']:<18} | {m['mae']:<6.2f} | {m['interval_mae']:<7.2f} | {m['rmse']:<6.2f} | {m['pcc']:<5.2f} | {m['spearman']:<8.2f} | {m['r2']:<6.2f} | {m['mcc']:<6.3f} | {m['f1']:<5.2f} | {m['roc_auc']:<5.2f} | {m['global_auc']:<10.3f} | {m['enrich']:<14.3f}")
    print("-" * 185)

    # Save all results to a single pt file for final plotting and papers
    out_results_path = os.path.join(PROJECT_ROOT, "new_data/fireprot_evaluation_results.pt")
    save_dict = {
        'y_true': y_true,
        'predictions': {name: data['y_pred'] for name, data in results.items()},
        'confidences': {name: data.get('y_conf', None) for name, data in results.items()},
        'metrics': metrics_summary
    }
    torch.save(save_dict, out_results_path)
    print(f"Saved evaluation results and metrics to {out_results_path}")

if __name__ == "__main__":
    main()
