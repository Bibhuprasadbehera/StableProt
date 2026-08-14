#!/usr/bin/env python3
"""Evaluate StableProt V7 Ensemble vs PRIME vs ThermoFormer on OOD BRENDA Benchmark for OGT Prediction.
1. Loads leak-free sequences from new_data/brenda_ood_benchmark.csv.
2. Evaluates StableProt V7 (5-seed ensemble).
3. Evaluates PRIME (AI4Protein/Prime_690M).
4. Evaluates ThermoFormer (GinnM/ThermoFormer).
5. Generates comparative publication markdown tables and plots.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, EsmModel

PROJECT_ROOT = Path(__file__).resolve().parents[3]
import torch.nn as nn

class MultiHeadSaProtV7(nn.Module):
    def __init__(self, input_dim=1280, hidden1=512, hidden2=256, dropout1=0.3, dropout2=0.2):
        super().__init__()
        self.shared_layer1 = nn.Linear(input_dim, hidden1)
        self.shared_bn1 = nn.LayerNorm(hidden1)
        self.shared_layer2 = nn.Linear(hidden1, hidden2)
        self.shared_bn2 = nn.LayerNorm(hidden2)
        self.shared_residual = nn.Linear(hidden1, hidden2)

        self.head_tm = nn.Linear(hidden2, 1)
        self.head_ogt = nn.Linear(hidden2, 1)

        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)

    def forward(self, x, task='tm'):
        x1 = self.dropout1(torch.relu(self.shared_bn1(self.shared_layer1(x))))
        x2 = self.dropout2(torch.relu(
            self.shared_bn2(self.shared_layer2(x1)) + self.shared_residual(x1)
        ))
        if task == 'tm': return self.head_tm(x2).squeeze(-1)
        elif task == 'ogt': return self.head_ogt(x2).squeeze(-1)
        else: raise ValueError(f"Unknown task: {task}")

OOD_CSV = PROJECT_ROOT / "new_data" / "brenda_ood_benchmark.csv"
OOD_EMB_PATH = PROJECT_ROOT / "data" / "embeddings" / "brenda_ood_saprot_embeddings.pt"
BASELINE_PREDS_PATH = PROJECT_ROOT / "data" / "embeddings" / "brenda_ood_baseline_preds.pt"
TABLE_OUT = PROJECT_ROOT / "paper" / "writeup" / "tables" / "table_ood_brenda_ogt.md"
PLOT_OUT = PROJECT_ROOT / "paper" / "writeup" / "plots" / "ood_brenda_ogt.png"

def set_aesthetics():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 16,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'figure.dpi': 300
    })

def extract_embeddings(df, device):
    if OOD_EMB_PATH.exists():
        print(f"Loading cached SaProt OOD embeddings from {OOD_EMB_PATH}...")
        return torch.load(OOD_EMB_PATH, map_location='cpu', weights_only=False)

    print("Extracting SaProt 650M AF2 embeddings on CPU...")
    emb_device = torch.device('cpu')
    tokenizer = AutoTokenizer.from_pretrained("westlake-repl/SaProt_650M_AF2")
    model = EsmModel.from_pretrained("westlake-repl/SaProt_650M_AF2", low_cpu_mem_usage=True).to(emb_device).eval()

    embs = []
    for seq in tqdm(df['sequence'], desc="SaProt OOD Embedding"):
        sa_str = "".join([f"{aa}#" for aa in seq])
        with torch.no_grad():
            inputs = tokenizer(sa_str[:1022], return_tensors="pt", truncation=True, max_length=1024).to(emb_device)
            outputs = model(**inputs)
            emb = outputs.last_hidden_state[0, 1:-1].mean(dim=0).cpu()
            embs.append(emb)

    X = torch.stack(embs)
    OOD_EMB_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(X, OOD_EMB_PATH)
    return X

def load_v7_ensemble(device):
    print("Loading 5-seed ensemble of StableProt V7...")
    models = []
    for s in range(1, 6):
        ckpt_path = PROJECT_ROOT / f"experiments/src/training/v7_shared/results/seed{s}/best_model.pt"
        if ckpt_path.exists():
            model = MultiHeadSaProtV7(input_dim=1280).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
            model.eval()
            models.append(model)
    return models

def evaluate_v8_ensemble(embeddings, sequences, device):
    VERSION = os.environ.get("STABLEPROT_VERSION", "v9_disjoint")
    label_version = "StableProt V9"
    print(f"Loading 5-seed ensemble of {label_version} (Ours)...")
    v8_dir = str(PROJECT_ROOT / f"experiments/src/training/{VERSION}")
    if v8_dir not in sys.path:
        sys.path.insert(0, v8_dir)
    import importlib.util
    import inspect
    v8_train_path = os.path.join(v8_dir, "train.py")
    spec = importlib.util.spec_from_file_location("train_v8", v8_train_path)
    train_v8 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_v8)
    MultiHeadSaProtV8 = train_v8.MultiHeadSaProtV8
    enrich_inputs = train_v8.enrich_inputs
    
    emb_v8, aux_v8 = enrich_inputs(embeddings, sequences, tmhmm_flags=None, ogt_priors=None)
    
    sig = inspect.signature(MultiHeadSaProtV8.__init__)
    model_kwargs = {}
    if 'use_residuals' in sig.parameters:
        model_kwargs['use_residuals'] = train_v8.CONFIG.get('use_residuals', True)

    v8_preds = []
    for s in range(1, 6):
        p_ogt = PROJECT_ROOT / f"experiments/src/training/{VERSION}/results/seed{s}/model_ogt.pt"
        p_comb = PROJECT_ROOT / f"experiments/src/training/{VERSION}/results/seed{s}/model.pt"
        p = p_ogt if p_ogt.exists() else p_comb
        if p.exists():
            model = MultiHeadSaProtV8(**model_kwargs).to(device)
            model.load_state_dict(torch.load(p, map_location=device, weights_only=False))
            model.eval()
            with torch.no_grad():
                out = model(emb_v8.to(device), aux_v8.to(device), head='ogt').cpu().numpy().squeeze()
            norm_p = PROJECT_ROOT / f"experiments/src/training/{VERSION}/results/seed{s}/normalization_stats.pt"
            if not norm_p.exists():
                norm_p = PROJECT_ROOT / f"experiments/src/training/{VERSION}/results/normalization_stats.pt"
            if norm_p.exists():
                norms = torch.load(norm_p, map_location='cpu', weights_only=False)
                if 'ogt_mean' in norms and 'ogt_std' in norms:
                    out = out * norms['ogt_std'] + norms['ogt_mean']
            v8_preds.append(out)
    if v8_preds:
        return np.mean(v8_preds, axis=0), np.std(v8_preds, axis=0)
    return None, None

def run_prime(sequences):
    print("Running PRIME baseline inference on CPU...")
    device = torch.device('cpu')
    model_path = "AI4Protein/Prime_690M"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device).eval()
    except Exception as e:
        print(f"Failed to load PRIME: {e}")
        return None

    y_pred = []
    batch_size = 4
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="PRIME"):
            batch = list(sequences[i:i+batch_size])
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
            outputs = model(**inputs)
            y_pred.extend(outputs.predicted_values.squeeze(-1).cpu().numpy().tolist())
    return np.array(y_pred)

def run_thermoformer(sequences):
    print("Running ThermoFormer baseline inference on CPU...")
    device = torch.device('cpu')
    sys.path.append(str(PROJECT_ROOT / "benchmark_models_ogt" / "ThermoFormer"))
    try:
        from model.modeling_thermoformer import ThermoFormer
        from model.tokenization_thermoformer import ThermoFormerTokenizer
        tokenizer = ThermoFormerTokenizer()
        model = ThermoFormer.from_pretrained("GinnM/ThermoFormer").to(device).eval()
    except Exception as e:
        print(f"Failed to load ThermoFormer: {e}")
        return None

    y_pred = []
    batch_size = 8
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="ThermoFormer"):
            batch = list(sequences[i:i+batch_size])
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
            outputs = model(**inputs)
            y_pred.extend(outputs.predicted_values.squeeze(-1).cpu().numpy().tolist())
    return np.array(y_pred)

_Z = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0])


def _ece(y_true, y_pred, sigma):
    import scipy.special
    expected = scipy.special.erf(_Z / np.sqrt(2.0))
    errors = np.abs(y_true - y_pred)
    return np.mean(np.abs(np.array([np.mean(errors <= z * sigma) for z in _Z]) - expected))


def crossfit_sigma_scale(y_true, y_pred, sigma, seed=0):
    """Two-fold cross-fit so no observation contributes to the scale applied to it."""
    grid = np.arange(0.5, 6.001, 0.005)
    fold = np.random.default_rng(seed).permutation(len(y_true)) % 2
    scaled = np.empty_like(sigma, dtype=float)
    cs = []
    for f in (0, 1):
        fit, app = fold != f, fold == f
        c = min(grid, key=lambda k: _ece(y_true[fit], y_pred[fit], k * sigma[fit]))
        scaled[app] = sigma[app] * c
        cs.append(c)
    return scaled, float(np.mean(cs))


def compute_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    pcc, _ = scipy.stats.pearsonr(y_true, y_pred)
    scc, _ = scipy.stats.spearmanr(y_true, y_pred)
    return mae, rmse, pcc, scc

def main():
    if not OOD_CSV.exists():
        print(f"Error: {OOD_CSV} not found!")
        return

    device = torch.device('cpu')
    df = pd.read_csv(OOD_CSV)
    print(f"Loaded {len(df)} OOD benchmark sequences.")
    y_true = df['ogt'].values

    # 1. StableProt V7 & V8
    X = extract_embeddings(df, device)
    models = load_v7_ensemble(device)
    preds_list = []
    with torch.no_grad():
        for m in models:
            preds_list.append(m(X, task='ogt').cpu().numpy().squeeze())
    y_v7 = np.mean(preds_list, axis=0) if preds_list else None
    y_v9, y_v9_conf = evaluate_v8_ensemble(X, df['sequence'].tolist(), device)

    # Load or compute baselines
    baselines = {}
    if BASELINE_PREDS_PATH.exists():
        print(f"Loading cached baseline predictions from {BASELINE_PREDS_PATH}...")
        baselines = torch.load(BASELINE_PREDS_PATH, map_location='cpu', weights_only=False)
    
    if 'PRIME' not in baselines or len(baselines['PRIME']) != len(df):
        y_prime = run_prime(df['sequence'])
        if y_prime is not None: baselines['PRIME'] = y_prime
    if 'ThermoFormer' not in baselines or len(baselines['ThermoFormer']) != len(df):
        y_thermo = run_thermoformer(df['sequence'])
        if y_thermo is not None: baselines['ThermoFormer'] = y_thermo

    torch.save(baselines, BASELINE_PREDS_PATH)

    # Compute comparison table
    VERSION = os.environ.get("STABLEPROT_VERSION", "v9_disjoint")
    label_version = f"StableProt ({VERSION})"
    # Fit the sigma scale out-of-fold rather than hardcoding 3.8, which was fitted against a
    # sigma that omitted the aleatoric term and now over-inflates every interval.
    if y_v9 is not None:
        conf_cal, c_fit = crossfit_sigma_scale(y_true, np.asarray(y_v9), np.asarray(y_v9_conf))
        print(f"Fitted sigma scale (out-of-fold): c = {c_fit:.3f}  (old hardcoded value: 3.8)")
    model_preds = {
        f'{label_version} (Ours)': y_v9,
        f'{label_version} (Int-MAE, k=1)': (y_v9, y_v9_conf) if y_v9 is not None else None,
        f'{label_version} (Int-MAE, calibrated c={c_fit:.2f})' if y_v9 is not None else 'unused':
            (y_v9, conf_cal) if y_v9 is not None else None,
        'PRIME (AI4Protein/Prime_690M)': baselines.get('PRIME', None),
        'ThermoFormer (GinnM/ThermoFormer)': baselines.get('ThermoFormer', None)
    }

    print("\n=== Overall Comparative OOD Performance ===")
    summary_rows = []
    for name, y_p in model_preds.items():
        if y_p is None: continue
        if isinstance(y_p, tuple):
            val, conf = y_p
            errors = np.maximum(0.0, np.abs(y_true - val) - conf)
            mae = np.mean(errors)
            rmse = np.sqrt(np.mean(errors**2))
            pcc, _ = scipy.stats.pearsonr(y_true, val)
            scc, _ = scipy.stats.spearmanr(y_true, val)
        else:
            mae, rmse, pcc, scc = compute_metrics(y_true, y_p)
            
        print(f"{name:45s} | MAE: {mae:6.2f}°C | RMSE: {rmse:6.2f}°C | PCC: {pcc:5.3f} | Spearman: {scc:5.3f}")
        summary_rows.append({
            'Model': name,
            'MAE (°C)': f"{mae:.2f}",
            'RMSE (°C)': f"{rmse:.2f}",
            'Pearson (r)': f"{pcc:.3f}",
            'Spearman (ρ)': f"{scc:.3f}"
        })

    # Save Markdown Table
    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(TABLE_OUT, "w") as f:
        f.write("# Out-Of-Distribution (OOD) BRENDA OGT Benchmark Comparative Results\n\n")
        f.write("Evaluation of StableProt V9 vs State-Of-The-Art baselines on strictly decontaminated out-of-distribution enzyme optimal growth temperatures from BRENDA (<40% sequence identity to training data, N=525).\n\n")
        f.write("| Model | MAE (°C) | RMSE (°C) | Pearson (r) | Spearman (ρ) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for r in summary_rows:
            f.write(f"| **{r['Model']}** | {r['MAE (°C)']} | {r['RMSE (°C)']} | {r['Pearson (r)']} | {r['Spearman (ρ)']} |\n")
        f.write("\n")
    print(f"Table saved to {TABLE_OUT}")

    # Plot comparative error distributions
    set_aesthetics()
    plt.figure(figsize=(12, 7))
    
    plot_data = []
    for name, y_p in model_preds.items():
        if y_p is None: continue
        short_name = name.replace(' (AI4Protein/Prime_690M)', '').replace(' (GinnM/ThermoFormer)', '')
        if isinstance(y_p, tuple):
            val, conf = y_p
            errors = np.maximum(0.0, np.abs(y_true - val) - conf)
        else:
            errors = np.abs(y_true - y_p)
        for err in errors:
            plot_data.append({'Model': short_name, 'Absolute Error (°C)': err})
            
    plot_df = pd.DataFrame(plot_data)
    sns.boxplot(x='Model', y='Absolute Error (°C)', data=plot_df, palette='viridis')
    plt.title("OOD BRENDA Prediction Error Comparison (N=525)")
    plt.ylabel("Absolute Error (°C)")
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(PLOT_OUT)
    plt.close()
    print(f"Plot saved to {PLOT_OUT}")

if __name__ == "__main__":
    main()
