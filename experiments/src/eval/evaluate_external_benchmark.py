#!/usr/bin/env python3
"""
Phase 6: Run PRIME and ThermoFormer inference on the newly created external OGT test set.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_TEST_FILE = PROJECT_ROOT / "new_data" / "external_ogt_benchmark.pt"
INTERNAL_TEST_FILE = PROJECT_ROOT / "data" / "embeddings" / "prepared_data_v7_saprot1.3b_seqonly_ogt_split.pt"
OUTPUT_FILE = PROJECT_ROOT / "experiments" / "src" / "eval" / "ogt_baselines" / "benchmark_predictions.pt"

# Ensure ThermoFormer is in path
sys.path.append(str(PROJECT_ROOT / "benchmark_models_ogt" / "ThermoFormer"))

from transformers import AutoTokenizer, AutoModel

def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    pcc, _ = pearsonr(y_true, y_pred)
    spearman, _ = spearmanr(y_true, y_pred)
    
    print(f"--- {name} ---")
    print(f"  MAE:      {mae:.4f}")
    print(f"  RMSE:     {rmse:.4f}")
    print(f"  PCC:      {pcc:.4f}")
    print(f"  Spearman: {spearman:.4f}\n")
    return {'mae': mae, 'rmse': rmse, 'pcc': pcc, 'spearman': spearman}

def run_prime(sequences, device):
    print(f"Loading PRIME model on {device}...")
    model_path = "AI4Protein/Prime_690M"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        model.eval()
        model.to(device)
    except Exception as e:
        print(f"Failed to load PRIME: {e}")
        return None

    y_pred = []
    batch_size = 16
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="PRIME"):
            batch_seqs = sequences[i:i+batch_size]
            inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            logits = outputs.predicted_values
            y_pred.extend(logits.squeeze(-1).cpu().numpy().tolist())
            
    # Clear memory
    del model
    del tokenizer
    torch.cuda.empty_cache()
    return np.array(y_pred)

def run_thermoformer(sequences, device):
    print(f"Loading ThermoFormer model on {device}...")
    try:
        from model.modeling_thermoformer import ThermoFormer
        from model.tokenization_thermoformer import ThermoFormerTokenizer
        tokenizer = ThermoFormerTokenizer()
        model = ThermoFormer.from_pretrained("GinnM/ThermoFormer")
        model.eval()
        model.to(device)
    except Exception as e:
        print(f"Failed to load ThermoFormer: {e}")
        return None

    y_pred = []
    batch_size = 16
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="ThermoFormer"):
            batch_seqs = sequences[i:i+batch_size]
            inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            logits = outputs.predicted_values
            y_pred.extend(logits.squeeze(-1).cpu().numpy().tolist())

    del model
    del tokenizer
    torch.cuda.empty_cache()
    return np.array(y_pred)

def plot_results(all_results):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Bar plot of MAE by Temp bins (10C intervals)
    bins = list(range(0, 110, 10))
    labels = [f'{bins[i]}-{bins[i+1]}°C' for i in range(len(bins)-1)]
    
    df_list = []
    for dataset_name, results_dict in all_results.items():
        y_true = results_dict['y_true']
        y_binned = pd.cut(y_true, bins=bins, labels=labels)
        for model_name, y_pred in results_dict.items():
            if model_name == 'y_true': continue
            if y_pred is None: continue
            errors = np.abs(y_true - y_pred)
            df_list.append(pd.DataFrame({
                'True_OGT': y_true, 
                'Error': errors, 
                'Model': model_name, 
                'Bin': y_binned,
                'Dataset': dataset_name
            }))
    
    df_all = pd.concat(df_list)
    
    # Plot External
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_all[df_all['Dataset'] == 'External'], x='Bin', y='Error', hue='Model', errorbar=None)
    plt.title('MAE across Temperature Ranges (External Benchmark)')
    plt.ylabel('Mean Absolute Error (°C)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE.parent / 'external_temp_wise_ogt.png', dpi=300)
    plt.close()
    
    # Plot Internal
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_all[df_all['Dataset'] == 'Internal'], x='Bin', y='Error', hue='Model', errorbar=None)
    plt.title('MAE across Temperature Ranges (Internal BacDive Test)')
    plt.ylabel('Mean Absolute Error (°C)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE.parent / 'internal_temp_wise_ogt.png', dpi=300)
    plt.close()

def main():
    if not EXTERNAL_TEST_FILE.exists():
        print(f"Error: {EXTERNAL_TEST_FILE} not found.")
        return
        
    print(f"Loading external test set: {EXTERNAL_TEST_FILE}")
    ext_data = torch.load(EXTERNAL_TEST_FILE, map_location='cpu')
    ext_seqs = ext_data['test_ogt']['sequences']
    ext_true = ext_data['test_ogt']['ogt_consensus'].numpy()
    
    print(f"Loading internal test set: {INTERNAL_TEST_FILE}")
    int_data = torch.load(INTERNAL_TEST_FILE, map_location='cpu', weights_only=False)
    # Randomly sample 5000 to match external set size
    int_seqs_all = int_data['test_ogt']['sequences']
    int_true_all = int_data['test_ogt']['ogt_consensus'].numpy()
    
    import random
    random.seed(42)
    indices = random.sample(range(len(int_seqs_all)), 5000)
    int_seqs = [int_seqs_all[i] for i in indices]
    int_true = np.array([int_true_all[i] for i in indices])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_results = {'External': {'y_true': ext_true}, 'Internal': {'y_true': int_true}}
    if OUTPUT_FILE.exists():
        all_results = torch.load(OUTPUT_FILE, map_location='cpu', weights_only=False)
    
    datasets = [('External', ext_seqs, ext_true), ('Internal', int_seqs, int_true)]
    
    for d_name, seqs, y_true in datasets:
        if d_name not in all_results:
            all_results[d_name] = {'y_true': y_true}
            
        # Run PRIME
        if 'PRIME' in all_results[d_name]:
            print(f"[{d_name}] Loaded existing PRIME predictions.")
            y_prime = all_results[d_name]['PRIME']
            evaluate(y_true, y_prime, f"PRIME ({d_name})")
        else:
            print(f"[{d_name}] Running PRIME...")
            y_prime = run_prime(seqs, device)
            if y_prime is not None:
                evaluate(y_true, y_prime, f"PRIME ({d_name})")
                all_results[d_name]['PRIME'] = y_prime
            
        # Run ThermoFormer
        if 'ThermoFormer' in all_results[d_name]:
            print(f"[{d_name}] Loaded existing ThermoFormer predictions.")
            y_thermo = all_results[d_name]['ThermoFormer']
            evaluate(y_true, y_thermo, f"ThermoFormer ({d_name})")
        else:
            print(f"[{d_name}] Running ThermoFormer...")
            y_thermo = run_thermoformer(seqs, device)
            if y_thermo is not None:
                evaluate(y_true, y_thermo, f"ThermoFormer ({d_name})")
                all_results[d_name]['ThermoFormer'] = y_thermo
            
        torch.save(all_results, OUTPUT_FILE)
        
    print(f"Predictions saved to {OUTPUT_FILE}")
    
    # Plotting
    plot_results(all_results)

if __name__ == "__main__":
    main()
