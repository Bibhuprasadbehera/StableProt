#!/usr/bin/env python3
"""
Extremophile Benchmark Evaluation Script.

Evaluates StableProt against literature baselines (TemBERTure, DeepSTABp, ESMStabP, ThermoFormer, PRIME)
specifically on extremophilic thermal regimes:
1. Stratified Continuous Metrics (MAE, CRPS, Pearson r, Spearman rho) across:
   - Psychrophiles (OGT < 20°C / Tm < 45°C)
   - Mesophile Controls (20-50°C / 45-65°C)
   - Moderate Thermophiles (50-65°C / 65-75°C)
   - Extreme Thermophiles (65-80°C / 75-85°C)
   - Hyperthermophiles (>80°C / >=85°C)
2. Extremophile Discrimination / Screening Metrics:
   - ROC AUC, PR-AUC, F1, Sensitivity, Specificity, Top-10% Enrichment Precision
3. Mesophilic Collapse Index:
   - Quantifies systematic under-prediction on extreme targets (mean signed error on thermophiles).
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr, norm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]

def crps_gaussian(y, mu, sigma):
    sigma = np.maximum(sigma, 1e-6)
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))

def compute_screening_metrics(y_true, y_pred, threshold=65.0, is_higher=True):
    """Compute binary classification metrics for extremophile screening."""
    if is_higher:
        binary_true = (y_true >= threshold).astype(int)
        scores = y_pred
    else:
        binary_true = (y_true <= threshold).astype(int)
        scores = -y_pred
        
    pos_count = np.sum(binary_true)
    if pos_count == 0 or pos_count == len(binary_true):
        return {'roc_auc': np.nan, 'pr_auc': np.nan, 'top10_precision': np.nan, 'f1': np.nan}
        
    roc_auc = float(roc_auc_score(binary_true, scores))
    pr_auc = float(average_precision_score(binary_true, scores))
    
    # Top 10% enrichment precision
    k = max(int(np.ceil(0.10 * len(scores))), 1)
    top_k_indices = np.argsort(scores)[::-1][:k]
    top10_prec = float(np.mean(binary_true[top_k_indices]))
    
    # Standard threshold prediction (predict positive if score >= threshold)
    binary_pred = (y_pred >= threshold).astype(int) if is_higher else (y_pred <= threshold).astype(int)
    f1 = float(f1_score(binary_true, binary_pred, zero_division=0))
    rec = float(recall_score(binary_true, binary_pred, zero_division=0))
    prec = float(precision_score(binary_true, binary_pred, zero_division=0))
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'top10_precision': top10_prec,
        'f1': f1,
        'sensitivity': rec,
        'precision': prec,
        'num_positives': int(pos_count),
        'total_samples': len(binary_true)
    }

def evaluate_regimes(y_true, predictions, regime_bins, tag="Benchmark"):
    """Evaluate all models across defined temperature regimes."""
    results = {}
    print(f"\n{'='*85}\n{tag} (Total Samples: {len(y_true)})\n{'='*85}")
    
    # Table Header
    hdr = f"{'Model':<20}" + "".join([f"{name:>13}" for name, _ in regime_bins]) + f"{'Overall MAE':>13}{'Pearson r':>11}"
    print(hdr)
    print("-" * len(hdr))
    
    for model_name, y_pred in predictions.items():
        err = np.abs(y_true - y_pred)
        overall_mae = np.mean(err)
        r_val = pearsonr(y_true, y_pred)[0] if len(y_true) > 2 else 0.0
        
        regime_maes = []
        model_regime_dict = {}
        
        for name, mask_fn in regime_bins:
            mask = mask_fn(y_true)
            cnt = np.sum(mask)
            if cnt > 0:
                bin_mae = np.mean(err[mask])
                regime_maes.append(f"{bin_mae:11.2f}°C")
                model_regime_dict[name] = float(bin_mae)
            else:
                regime_maes.append(f"{'N/A':>13}")
                model_regime_dict[name] = np.nan
                
        line = f"{model_name:<20}" + "".join(regime_maes) + f"{overall_mae:11.2f}°C{r_val:11.3f}"
        print(line)
        
        # Mesophilic collapse score (signed bias on thermophiles)
        thermo_mask = y_true >= 65.0
        signed_bias = float(np.mean(y_true[thermo_mask] - y_pred[thermo_mask])) if np.sum(thermo_mask) > 0 else 0.0
        
        results[model_name] = {
            'overall_mae': float(overall_mae),
            'pearson_r': float(r_val),
            'spearman_rho': float(spearmanr(y_true, y_pred)[0]) if len(y_true) > 2 else 0.0,
            'regime_maes': model_regime_dict,
            'thermo_underprediction_bias': signed_bias
        }
        
    return results

def evaluate_tm_extremophiles(pt_path):
    """Evaluate Tm models on ProThermDB and FireProtDB extremophiles."""
    print("\n" + "#"*85 + "\nEvaluating Melting Temperature (Tm) Extremophiles\n" + "#"*85)
    
    # 1. ProThermDB
    protherm_path = PROJECT_ROOT / "new_data/protherm_evaluation_results.pt"
    canonical_models = ['StableProt V9', 'StableProt', 'TemBERTure', 'ESMStabP', 'DeepSTABp', 'ThermoFormer', 'TemStaPro']
    if protherm_path.exists():
        data = torch.load(protherm_path, map_location='cpu', weights_only=False)
        y_true = np.array(data['y_true'])
        preds = {}
        for k, v in data['predictions'].items():
            if k in canonical_models:
                clean_k = 'StableProt (Ours)' if 'StableProt' in k else k
                if clean_k not in preds:
                    preds[clean_k] = np.array(v)
            
        tm_bins = [
            ('< 50°C (Meso-Low)', lambda t: t < 50.0),
            ('50-65°C (Meso)', lambda t: (t >= 50.0) & (t < 65.0)),
            ('65-80°C (Thermo)', lambda t: (t >= 65.0) & (t < 80.0)),
            ('>= 80°C (Hyper)', lambda t: t >= 80.0),
            ('>= 90°C (Extreme)', lambda t: t >= 90.0)
        ]
        
        res_pt = evaluate_regimes(y_true, preds, tm_bins, tag="ProThermDB Validation (n=3,340)")
        
        # Screening Table
        print("\n--- ProThermDB Extremophile Screening Performance ---")
        for cutoff in [65.0, 80.0]:
            print(f"\nTarget Threshold Tm >= {cutoff:.0f}°C:")
            print(f"{'Model':<20}{'ROC AUC':>10}{'PR AUC':>10}{'Top-10% Prec':>15}{'F1':>8}{'Sensitivity':>13}")
            print("-" * 76)
            for m_name, y_p in preds.items():
                m = compute_screening_metrics(y_true, y_p, threshold=cutoff)
                print(f"{m_name:<20}{m['roc_auc']:10.3f}{m['pr_auc']:10.3f}{m['top10_precision']:15.3f}{m['f1']:8.3f}{m['sensitivity']:13.3f}")

    # 2. FireProtDB Zero-Shot Holdout
    fireprot_path = PROJECT_ROOT / "new_data/fireprot_evaluation_results.pt"
    if fireprot_path.exists():
        data = torch.load(fireprot_path, map_location='cpu', weights_only=False)
        y_true = np.array(data['y_true'])
        preds = {}
        for k, v in data['predictions'].items():
            if k in canonical_models:
                clean_k = 'StableProt (Ours)' if 'StableProt' in k else k
                if clean_k not in preds:
                    preds[clean_k] = np.array(v)
            
        fp_bins = [
            ('< 45°C (Psychro-labile)', lambda t: t < 45.0),
            ('45-65°C (Meso)', lambda t: (t >= 45.0) & (t < 65.0)),
            ('65-80°C (Thermo)', lambda t: (t >= 65.0) & (t < 80.0)),
            ('>= 80°C (Hyperthermo)', lambda t: t >= 80.0)
        ]
        
        res_fp = evaluate_regimes(y_true, preds, fp_bins, tag="FireProtDB Zero-Shot Holdout (n=322)")
        
        # Screening Table
        print("\n--- FireProtDB Zero-Shot Extremophile Screening Performance ---")
        for cutoff in [65.0, 80.0]:
            print(f"\nTarget Threshold Tm >= {cutoff:.0f}°C:")
            print(f"{'Model':<20}{'ROC AUC':>10}{'PR AUC':>10}{'Top-10% Prec':>15}{'F1':>8}{'Sensitivity':>13}")
            print("-" * 76)
            for m_name, y_p in preds.items():
                m = compute_screening_metrics(y_true, y_p, threshold=cutoff)
                print(f"{m_name:<20}{m['roc_auc']:10.3f}{m['pr_auc']:10.3f}{m['top10_precision']:15.3f}{m['f1']:8.3f}{m['sensitivity']:13.3f}")

def evaluate_ogt_extremophiles(pt_path):
    """Evaluate OGT models on BRENDA OOD benchmark extremophiles."""
    print("\n" + "#"*85 + "\nEvaluating Optimal Growth Temperature (OGT) Extremophiles\n" + "#"*85)
    
    cache_path = PROJECT_ROOT / "paper/writeup/plots/_cache_brenda_ogt.npz"
    if cache_path.exists():
        cache = np.load(cache_path)
        y_true = cache['y_true']
        preds = {
            'StableProt (Ours)': cache['y_pred']
        }
        for k in cache.files:
            if k.startswith("pred_") and k not in ["pred_StableProt", "pred_StableProt V9"]:
                name = k.replace("pred_", "")
                preds[name] = cache[k]
                
        ogt_bins = [
            ('< 20°C (Psychrophile)', lambda t: t < 20.0),
            ('20-50°C (Mesophile)', lambda t: (t >= 20.0) & (t <= 50.0)),
            ('50-65°C (Moderate Thermo)', lambda t: (t > 50.0) & (t <= 65.0)),
            ('65-80°C (Extreme Thermo)', lambda t: (t > 65.0) & (t <= 80.0)),
            ('> 80°C (Hyperthermo)', lambda t: t > 80.0)
        ]
        
        evaluate_regimes(y_true, preds, ogt_bins, tag="BRENDA OOD Benchmark (n=525)")
        
        # Screening Table
        print("\n--- BRENDA OOD Extremophile Screening Performance ---")
        for cutoff, name in [(20.0, "Psychrophile (OGT <= 20°C)"), (50.0, "Thermophile (OGT >= 50°C)"), (65.0, "Extreme Thermophile (OGT >= 65°C)")]:
            is_higher = cutoff >= 50.0
            print(f"\nTarget Regime: {name}")
            print(f"{'Model':<20}{'ROC AUC':>10}{'PR AUC':>10}{'Top-10% Prec':>15}{'F1':>8}{'Sensitivity':>13}")
            print("-" * 76)
            for m_name, y_p in preds.items():
                m = compute_screening_metrics(y_true, y_p, threshold=cutoff, is_higher=is_higher)
                print(f"{m_name:<20}{m['roc_auc']:10.3f}{m['pr_auc']:10.3f}{m['top10_precision']:15.3f}{m['f1']:8.3f}{m['sensitivity']:13.3f}")

def main():
    bench_pt_path = PROJECT_ROOT / "data/extremophile_dataset/pt/extremophile_benchmark.pt"
    evaluate_tm_extremophiles(bench_pt_path)
    evaluate_ogt_extremophiles(bench_pt_path)

if __name__ == "__main__":
    main()
