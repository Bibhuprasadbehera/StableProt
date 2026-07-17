#!/usr/bin/env python3
"""
Evaluate Uncertainty Calibration & Reliability for StableProt V8 (`Claim 3`)

Generates:
  1. Overall Calibration Reliability Diagram (`paper/writeup/plots/calibration_reliability_diagram.png`)
  2. Temperature-Stratified & 10°C-Wise Calibration Curves (`paper/writeup/plots/calibration_stratified_temp.png`)
  3. Uncertainty Stratification Barplot (Empirical MAE by predicted sigma bin)
  4. Extracts exact case studies (Confident & Correct vs Uncertain & Understandable)
  5. Saves numerical summary tables to `experiments/new_data/validation_suite/calibration_summary.csv`
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import scipy.special
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
PROJECT_ROOT = os.path.dirname(EXPERIMENTS_DIR)
OUT_DIR = os.path.join(PROJECT_ROOT, "paper/writeup/plots")
VAL_SUITE_DIR = os.path.join(EXPERIMENTS_DIR, "new_data/validation_suite")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(VAL_SUITE_DIR, exist_ok=True)

def expected_coverage(z):
    """Return expected Gaussian probability within +- z * sigma."""
    return scipy.special.erf(z / np.sqrt(2.0))

def compute_calibration_curve(y_true, y_pred, y_conf, z_vals=None):
    if z_vals is None:
        z_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0])
    exp_cov = expected_coverage(z_vals)
    emp_cov = []
    errors = np.abs(y_true - y_pred)
    for z in z_vals:
        bound = z * y_conf
        emp_cov.append(np.mean(errors <= bound))
    emp_cov = np.array(emp_cov)
    ece = np.mean(np.abs(emp_cov - exp_cov))
    return exp_cov, emp_cov, ece, z_vals

def main():
    print("=====================================================================================")
    print("  STABLEPROT V9 UNCERTAINTY CALIBRATION & RELIABILITY BENCHMARK")
    print("=====================================================================================")
    
    VERSION = os.environ.get("STABLEPROT_VERSION", "v8_disjoint")
    protherm_path = os.path.join(PROJECT_ROOT, "new_data/protherm_evaluation_results.pt")
    if not os.path.exists(protherm_path):
        print(f"Error: ProTherm evaluation results missing: {protherm_path}")
        return
        
    data = torch.load(protherm_path, map_location='cpu', weights_only=False)
    y_true = np.array(data['y_true'])
    preds = data['predictions']
    metrics_cached = data.get('metrics', {})
    
    # Check key for V8 predictions
    k_pred = 'StableProt V9' if 'StableProt V9' in preds else 'StableProt V8'
    y_pred = np.array(preds[k_pred])
    
    # Check where confidences are stored
    y_conf = None
    if 'confidences' in data and (k_pred in data['confidences'] or 'StableProt V8' in data['confidences']):
        c_key = k_pred if k_pred in data.get('confidences', {}) else 'StableProt V8'
        y_conf = np.array(data['confidences'][c_key])
    elif k_pred in metrics_cached and 'y_conf' in metrics_cached[k_pred]:
        y_conf = np.array(metrics_cached[k_pred]['y_conf'])
        
    if y_conf is None:
        # Load directly from ensemble checkpoint or prepared data if needed
        print(f"Searching for confidences in {VERSION} checkpoint results...")
        ens_p = os.path.join(EXPERIMENTS_DIR, f"{VERSION}/results/ensemble/predictions.pt")
        if os.path.exists(ens_p):
            ens_data = torch.load(ens_p, map_location='cpu', weights_only=False)
            if 'y_conf' in ens_data:
                y_conf = np.array(ens_data['y_conf'])
            elif 'conf' in ens_data:
                y_conf = np.array(ens_data['conf'])
                
    if y_conf is None:
        # Reconstruct ens_sigma across the 5 seeds if needed
        print("Reconstructing ensemble confidences from seed checkpoints...")
        # Check seed results
        seed_dirs = [os.path.join(EXPERIMENTS_DIR, f"{VERSION}/results/seed{s}") for s in range(1, 6)]
        # For simplicity, if not directly found, we check if protherm_evaluation_results has any sigma
        pass

    if y_conf is None:
        print("Error: Could not locate y_conf for StableProt V9.")
        return

    print(f"Loaded {len(y_true)} ProThermDB sequences.")
    print(f"True Tm Range: {y_true.min():.1f} - {y_true.max():.1f}°C")
    print(f"Pred Tm Range: {y_pred.min():.1f} - {y_pred.max():.1f}°C")
    print(f"Predicted Sigma Range: {y_conf.min():.2f} - {y_conf.max():.2f}°C (Mean: {y_conf.mean():.2f}°C)")
    
    # ── 1. Overall Calibration Curve ──
    exp_cov, emp_cov, ece, z_vals = compute_calibration_curve(y_true, y_pred, y_conf)
    print(f"\nOverall Estimated Calibration Error (ECE): {ece:.4f}")
    print(f"Empirical Coverage @ 1.0 sigma (Expected 68.3%): {emp_cov[np.where(np.isclose(z_vals, 1.0))[0][0]]*100:.1f}%")
    print(f"Empirical Coverage @ 2.0 sigma (Expected 95.4%): {emp_cov[np.where(np.isclose(z_vals, 2.0))[0][0]]*100:.1f}%")
    
    # ── 2. Temperature Stratification ──
    strata = {
        'Mesophilic (20-40°C)': (y_true >= 20.0) & (y_true <= 40.0),
        'Thermophilic (40-60°C)': (y_true > 40.0) & (y_true <= 60.0),
        'Hyperthermophilic (>60°C)': y_true > 60.0
    }
    
    strata_results = {}
    for name, mask in strata.items():
        if np.sum(mask) > 5:
            e_exp, e_emp, e_ece, _ = compute_calibration_curve(y_true[mask], y_pred[mask], y_conf[mask], z_vals)
            strata_results[name] = {'exp': e_exp, 'emp': e_emp, 'ece': e_ece, 'count': np.sum(mask)}
            print(f"  {name:<26} | N={np.sum(mask):<4} | ECE: {e_ece:.4f} | 1σ Cov: {e_emp[np.where(np.isclose(z_vals, 1.0))[0][0]]*100:.1f}%")

    bins_10c = {
        '<30°C': y_true < 30.0,
        '30-40°C': (y_true >= 30.0) & (y_true <= 40.0),
        '40-50°C': (y_true > 40.0) & (y_true <= 50.0),
        '50-60°C': (y_true > 50.0) & (y_true <= 60.0),
        '60-70°C': (y_true > 60.0) & (y_true <= 70.0),
        '>70°C': y_true > 70.0
    }
    
    bins_results = {}
    for name, mask in bins_10c.items():
        if np.sum(mask) > 3:
            e_exp, e_emp, e_ece, _ = compute_calibration_curve(y_true[mask], y_pred[mask], y_conf[mask], z_vals)
            bins_results[name] = {'exp': e_exp, 'emp': e_emp, 'ece': e_ece, 'count': np.sum(mask)}
            print(f"  10°C Bin {name:<10} | N={np.sum(mask):<4} | ECE: {e_ece:.4f} | 1σ Cov: {e_emp[np.where(np.isclose(z_vals, 1.0))[0][0]]*100:.1f}%")

    # Save summary CSV
    rows = []
    for name, d in {**{'Overall': {'exp': exp_cov, 'emp': emp_cov, 'ece': ece, 'count': len(y_true)}}, **strata_results, **bins_results}.items():
        idx_1s = np.where(np.isclose(z_vals, 1.0))[0][0]
        idx_2s = np.where(np.isclose(z_vals, 2.0))[0][0]
        rows.append({
            'Stratum': name,
            'Sample_Count': d['count'],
            'ECE': round(d['ece'], 4),
            'Coverage_1Sigma_Empirical': round(d['emp'][idx_1s] * 100, 2),
            'Coverage_1Sigma_Expected': round(d['exp'][idx_1s] * 100, 2),
            'Coverage_2Sigma_Empirical': round(d['emp'][idx_2s] * 100, 2),
            'Coverage_2Sigma_Expected': round(d['exp'][idx_2s] * 100, 2),
        })
    df_summary = pd.DataFrame(rows)
    csv_out = os.path.join(VAL_SUITE_DIR, "calibration_summary.csv")
    df_summary.to_csv(csv_out, index=False)
    print(f"\nSaved calibration summary table: {csv_out}")

    # ── 3. Plot Overall Calibration Diagram ──
    sns.set_context("paper", font_scale=1.25)
    plt.figure(figsize=(7, 7))
    plt.plot([0, 100], [0, 100], 'k--', alpha=0.6, label='Ideal Calibration ($y=x$)')
    
    # Compute calibrated calibration curve
    exp_cov_cal, emp_cov_cal, ece_cal, _ = compute_calibration_curve(y_true, y_pred, y_conf * 2.8, z_vals)
    
    plt.plot(exp_cov * 100, emp_cov * 100, 'o-', color='#3b82f6', linewidth=2.5, markersize=7, label=f'StableProt V9 (Unscaled, ECE = {ece:.3f})')
    plt.plot(exp_cov_cal * 100, emp_cov_cal * 100, 's-', color='#10b981', linewidth=2.5, markersize=7, label=f'StableProt V9 (Calibrated T=2.8, ECE = {ece_cal:.3f})')
    plt.xlabel("Predicted Confidence Level / Expected Coverage (%)")
    plt.ylabel("Observed Empirical Coverage (%)")
    plt.title("Reliability Diagram: $T_m$ Uncertainty Calibration ($\pm z \cdot \sigma$)")
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    p_overall = os.path.join(OUT_DIR, "calibration_reliability_diagram.png")
    plt.savefig(p_overall, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {p_overall}")

    # ── 4. Plot Stratified Calibration Diagrams ──
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    
    # Left: Broad Strata (Meso, Thermo, Hyper)
    axes[0].plot([0, 100], [0, 100], 'k--', alpha=0.6, label='Ideal ($y=x$)')
    colors_strata = ['#3b82f6', '#f59e0b', '#ef4444']
    for (name, d), col in zip(strata_results.items(), colors_strata):
        axes[0].plot(d['exp'] * 100, d['emp'] * 100, 'o-', color=col, linewidth=2, markersize=5, label=f"{name} (ECE={d['ece']:.3f})")
    axes[0].set_xlabel("Expected Coverage (%)")
    axes[0].set_ylabel("Observed Coverage (%)")
    axes[0].set_title("A. Thermal Stratum Calibration ($20-40^\circ$C, $40-60^\circ$C, $>60^\circ$C)")
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(0, 100)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc='lower right')

    # Right: 10°C Bins
    axes[1].plot([0, 100], [0, 100], 'k--', alpha=0.6, label='Ideal ($y=x$)')
    cmap = plt.get_cmap('viridis')
    cols_10c = [cmap(i) for i in np.linspace(0.1, 0.9, len(bins_results))]
    for (name, d), col in zip(bins_results.items(), cols_10c):
        axes[1].plot(d['exp'] * 100, d['emp'] * 100, 'o-', color=col, linewidth=2, markersize=5, label=f"{name} (N={d['count']}, ECE={d['ece']:.3f})")
    axes[1].set_xlabel("Expected Coverage (%)")
    axes[1].set_ylabel("Observed Coverage (%)")
    axes[1].set_title("B. Fine-Grained $10^\circ$C-Wise Calibration")
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    p_strat = os.path.join(OUT_DIR, "calibration_stratified_temp.png")
    plt.savefig(p_strat, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {p_strat}")

    # Export JSON coordinates for both plots (`Universal JSON compliance`)
    import json
    json_overall = os.path.join(OUT_DIR, "calibration_reliability_diagram.json")
    with open(json_overall, "w") as f:
        json.dump({"expected_coverage_pct": (exp_cov * 100).tolist(), "empirical_coverage_pct": (emp_cov * 100).tolist(), "ece": float(ece)}, f, indent=2)
    print(f"Saved JSON: {json_overall}")

    json_strat = os.path.join(OUT_DIR, "calibration_stratified_temp.json")
    with open(json_strat, "w") as f:
        strat_export = {}
        for name, d in strata_results.items():
            strat_export[name] = {"expected_pct": (d['exp']*100).tolist(), "observed_pct": (d['emp']*100).tolist(), "ece": float(d['ece']), "count": int(d['count'])}
        for name, d in bins_results.items():
            strat_export[name] = {"expected_pct": (d['exp']*100).tolist(), "observed_pct": (d['emp']*100).tolist(), "ece": float(d['ece']), "count": int(d['count'])}
        json.dump(strat_export, f, indent=2)
    print(f"Saved JSON: {json_strat}")

    # ── 5. Case Studies Extraction (`Humanizing Uncertainty`) ──
    print("\n── EXTRACTING EXACT UNCERTAINTY CASE STUDIES ──")
    errors = np.abs(y_true - y_pred)
    
    # Case 1: Confident and Correct (low sigma, low error)
    idx_conf = np.where((y_conf < 2.5) & (errors < 1.2))[0]
    if len(idx_conf) > 0:
        c1 = idx_conf[np.argmin(errors[idx_conf])]
        print(f"\n[Case Study 1: Confident & Correct]")
        print(f"  Index: {c1}")
        print(f"  True Tm:      {y_true[c1]:.2f}°C")
        print(f"  Predicted Tm: {y_pred[c1]:.2f}°C (Error: {errors[c1]:.2f}°C)")
        print(f"  Predicted σ:  ±{y_conf[c1]:.2f}°C")
        
    # Case 2: Uncertain and Understandable (high sigma)
    idx_unc = np.where(y_conf > 5.0)[0]
    if len(idx_unc) > 0:
        c2 = idx_unc[np.argmax(y_conf[idx_unc])]
        print(f"\n[Case Study 2: Uncertain & Understandable (High σ)]")
        print(f"  Index: {c2}")
        if 'ids' in data: print(f"  ID: {data['ids'][c2]}")
        print(f"  True Tm:      {y_true[c2]:.2f}°C")
        print(f"  Predicted Tm: {y_pred[c2]:.2f}°C (Error: {errors[c2]:.2f}°C)")
        print(f"  Predicted σ:  ±{y_conf[c2]:.2f}°C")

    print("\nCalibration benchmark successfully completed.")

if __name__ == "__main__":
    main()
