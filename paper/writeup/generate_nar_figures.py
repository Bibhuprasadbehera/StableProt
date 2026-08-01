#!/usr/bin/env python3
"""Generate all NAR-ready figures for StableProt manuscript."""
import os, sys, json
import numpy as np
import pandas as pd
import torch
import scipy.special
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Global Style ──
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.15, 'grid.linestyle': '-',
    'figure.dpi': 150, 'savefig.dpi': 600, 'savefig.bbox': 'tight',
})

# Colorblind-safe palette
COLORS = {
    'StableProt': '#0077B6', 'StableProt_cal': '#00B4D8',
    'TemBERTure': '#E69F00', 'DeepSTABp': '#CC79A7',
    'ESMStabP': '#56B4E9', 'TemStaPro': '#009E73',
    'ThermoFormer': '#D55E00', 'PRIME': '#F0E442',
    'Composition': '#999999', 'SaProt': '#6C757D',
}

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(PROJECT, "paper/writeup/plots")
os.makedirs(OUT, exist_ok=True)

def expected_coverage(z):
    return scipy.special.erf(z / np.sqrt(2.0))

# ════════════════════════════════════════════════════════════════════
# 1. CALIBRATION RELIABILITY DIAGRAM (T=3.8)
# ════════════════════════════════════════════════════════════════════
def plot_calibration():
    print("── 1. Calibration Reliability Diagram (T=3.8) ──")
    pt_path = os.path.join(PROJECT, "new_data/protherm_evaluation_results.pt")
    if not os.path.exists(pt_path):
        print(f"SKIP: {pt_path} missing"); return
    
    data = torch.load(pt_path, map_location='cpu', weights_only=False)
    y_true = np.array(data['y_true'])
    k = 'StableProt V9' if 'StableProt V9' in data['predictions'] else 'StableProt V8'
    y_pred = np.array(data['predictions'][k])
    
    y_conf = None
    if 'confidences' in data:
        ck = k if k in data['confidences'] else 'StableProt V8'
        if ck in data['confidences']:
            y_conf = np.array(data['confidences'][ck])
    if y_conf is None:
        print("SKIP: no confidence data"); return
    
    z_vals = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.4,1.6,1.8,2.0,2.5,3.0])
    errors = np.abs(y_true - y_pred)
    
    def cal_curve(conf):
        exp_c = expected_coverage(z_vals)
        emp_c = np.array([np.mean(errors <= z * conf) for z in z_vals])
        ece = np.mean(np.abs(emp_c - exp_c))
        return exp_c, emp_c, ece
    
    exp_raw, emp_raw, ece_raw = cal_curve(y_conf)
    exp_cal, emp_cal, ece_cal = cal_curve(y_conf * 3.8)  # T=3.8!
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0,100],[0,100], '--', color='#888', alpha=0.7, lw=1.5, label='Ideal ($y=x$)')
    ax.plot(exp_raw*100, emp_raw*100, 'o-', color='#3b82f6', lw=2, ms=6, 
            label=f'Unscaled (ECE = {ece_raw:.3f})')
    ax.plot(exp_cal*100, emp_cal*100, 's-', color='#10b981', lw=2, ms=6,
            label=f'Calibrated $T$=3.8 (ECE = {ece_cal:.3f})')
    ax.set_xlabel("Expected Coverage (%)")
    ax.set_ylabel("Observed Coverage (%)")
    ax.set_title("Reliability Diagram: $T_m$ Uncertainty Calibration")
    ax.set_xlim(0,100); ax.set_ylim(0,100)
    ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='#ddd')
    
    out_path = os.path.join(OUT, "calibration_reliability_diagram.png")
    fig.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")
    print(f"  ECE raw={ece_raw:.4f}, ECE T=3.8={ece_cal:.4f}")
    return y_true, y_pred, y_conf

# ════════════════════════════════════════════════════════════════════
# 2. CONFIDENCE SPREAD VISUALIZATION
# ════════════════════════════════════════════════════════════════════
def plot_confidence_spread(y_true, y_pred, y_conf):
    print("── 2. Confidence Spread Visualization ──")
    errors = np.abs(y_true - y_pred)
    sort_idx = np.argsort(y_conf)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Panel A: Sorted predictions with confidence bands
    ax = axes[0]
    n = len(y_true)
    x = np.arange(n)
    ax.fill_between(x, y_pred[sort_idx] - y_conf[sort_idx], 
                    y_pred[sort_idx] + y_conf[sort_idx],
                    alpha=0.25, color=COLORS['StableProt'], label='±1σ band')
    ax.fill_between(x, y_pred[sort_idx] - 2*y_conf[sort_idx],
                    y_pred[sort_idx] + 2*y_conf[sort_idx],
                    alpha=0.1, color=COLORS['StableProt_cal'], label='±2σ band')
    ax.scatter(x, y_true[sort_idx], s=3, c='#e74c3c', alpha=0.6, zorder=5, label='True $T_m$')
    ax.scatter(x, y_pred[sort_idx], s=2, c=COLORS['StableProt'], alpha=0.4, zorder=4)
    ax.set_xlabel("Proteins (sorted by predicted σ)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("(A) Predictions with Uncertainty Bands")
    ax.legend(loc='upper left', markerscale=3)
    
    # Panel B: Histogram of predicted sigma
    ax = axes[1]
    ax.hist(y_conf, bins=40, color=COLORS['StableProt'], alpha=0.7, edgecolor='white', lw=0.5)
    median_s = np.median(y_conf)
    p25, p75 = np.percentile(y_conf, [25, 75])
    ax.axvline(median_s, color='#e74c3c', ls='--', lw=2, label=f'Median σ = {median_s:.1f}°C')
    ax.axvspan(p25, p75, alpha=0.15, color='#f39c12', label=f'IQR [{p25:.1f}, {p75:.1f}]°C')
    ax.set_xlabel("Predicted Uncertainty σ (°C)")
    ax.set_ylabel("Count")
    ax.set_title("(B) Uncertainty Distribution")
    ax.legend(loc='upper right')
    
    # Panel C: Error vs Confidence scatter (is model calibrated?)
    ax = axes[2]
    # Bin by sigma quintiles
    q_edges = np.percentile(y_conf, [0, 20, 40, 60, 80, 100])
    bin_labels = ['Very Low σ', 'Low σ', 'Medium σ', 'High σ', 'Very High σ']
    bin_colors = ['#2ecc71', '#27ae60', '#f39c12', '#e67e22', '#e74c3c']
    mae_bins, sigma_bins = [], []
    for i in range(5):
        mask = (y_conf >= q_edges[i]) & (y_conf < q_edges[i+1] + 0.01)
        if mask.sum() > 0:
            mae_bins.append(errors[mask].mean())
            sigma_bins.append(y_conf[mask].mean())
    
    bars = ax.bar(bin_labels, mae_bins, color=bin_colors, edgecolor='white', lw=1)
    # Overlay mean sigma as line
    ax2 = ax.twinx()
    ax2.plot(bin_labels, sigma_bins, 'D-', color='#2c3e50', lw=2, ms=8, label='Mean σ')
    ax2.set_ylabel("Mean Predicted σ (°C)", color='#2c3e50')
    ax2.spines['right'].set_visible(True)
    ax2.spines['top'].set_visible(False)
    ax.set_ylabel("Mean Absolute Error (°C)")
    ax.set_title("(C) Error Scales with Uncertainty")
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    out_path = os.path.join(OUT, "confidence_spread_analysis.png")
    fig.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")

# ════════════════════════════════════════════════════════════════════
# 3. THERMOFORMER/PRIME OVERFITTING SUBPLOT
# ════════════════════════════════════════════════════════════════════
def plot_overfitting_subplot():
    print("── 3. ThermoFormer/PRIME Overfitting Subplot ──")
    # Data from Table 5 in manuscript
    bins = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']
    sp_raw = [29.10, 21.14, 13.45, 11.24, 7.96, 10.09, 11.11, 8.88, 7.87, 4.76]
    sp_cal = [11.81, 7.63, 2.97, 2.34, 1.54, 3.07, 3.73, 2.03, 0.70, 0.21]
    prime  = [20.86, 8.80, 3.68, 2.26, 12.72, 12.51, 7.56, 5.30, 6.67, 5.25]
    thermo = [21.03, 8.47, 3.38, 2.35, 11.48, 12.19, 6.91, 5.27, 6.49, 5.29]
    
    fig, (ax_main, ax_inset) = plt.subplots(1, 2, figsize=(14, 5.5),
                                             gridspec_kw={'width_ratios': [3, 1.2]})
    
    x = np.arange(len(bins))
    w = 0.22
    
    ax_main.bar(x - w, sp_cal, w, color=COLORS['StableProt_cal'], label='StableProt V9 (Calibrated)', edgecolor='white', lw=0.5)
    ax_main.bar(x, prime, w, color=COLORS['PRIME'], label='PRIME', edgecolor='#888', lw=0.5)
    ax_main.bar(x + w, thermo, w, color=COLORS['ThermoFormer'], label='ThermoFormer', edgecolor='white', lw=0.5)
    
    # Highlight overfitting zone
    ax_main.axvspan(3.5, 5.5, alpha=0.08, color='red', zorder=0)
    ax_main.annotate('Overfitting\nCollapse Zone', xy=(4.5, 13.5), fontsize=9, 
                     ha='center', color='#c0392b', fontstyle='italic', fontweight='bold')
    
    ax_main.set_xticks(x)
    ax_main.set_xticklabels([f'{b}°C' for b in bins], rotation=30, ha='right')
    ax_main.set_ylabel("MAE (°C)")
    ax_main.set_title("Per-Bin OGT Error: Mesophilic Overfitting in PRIME & ThermoFormer")
    ax_main.legend(loc='upper left')
    ax_main.set_ylim(0, 15)
    
    # Inset: Error ratio (40-60°C) / (20-40°C) 
    meso_mae = {'StableProt': np.mean(sp_cal[2:4]), 'PRIME': np.mean(prime[2:4]), 'ThermoFormer': np.mean(thermo[2:4])}
    trans_mae = {'StableProt': np.mean(sp_cal[4:6]), 'PRIME': np.mean(prime[4:6]), 'ThermoFormer': np.mean(thermo[4:6])}
    
    models = list(meso_mae.keys())
    ratios = [trans_mae[m] / max(meso_mae[m], 0.01) for m in models]
    bar_colors = [COLORS['StableProt_cal'], COLORS['PRIME'], COLORS['ThermoFormer']]
    
    bars = ax_inset.bar(models, ratios, color=bar_colors, edgecolor='white', lw=1)
    for bar, r in zip(bars, ratios):
        ax_inset.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                      f'{r:.1f}×', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax_inset.axhline(1.0, color='#888', ls='--', lw=1, alpha=0.5)
    ax_inset.set_ylabel("Error Ratio: (40-60°C) / (20-40°C)")
    ax_inset.set_title("Overfitting Ratio", fontweight='bold')
    ax_inset.set_ylim(0, max(ratios) + 1)
    
    plt.tight_layout()
    out_path = os.path.join(OUT, "overfitting_subplot_prime_thermoformer.png")
    fig.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")

# ════════════════════════════════════════════════════════════════════
# 4. ARCHITECTURE BOILERPLATE (text-based flowchart)
# ════════════════════════════════════════════════════════════════════
def plot_architecture_boilerplate():
    print("── 4. Architecture Boilerplate ──")
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14); ax.set_ylim(0, 10)
    ax.axis('off')
    
    def box(x, y, w, h, text, color='#0077B6', alpha=0.15, fs=9, bold=False):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                        facecolor=color, alpha=alpha, edgecolor=color, lw=1.5)
        ax.add_patch(rect)
        weight = 'bold' if bold else 'normal'
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fs, fontweight=weight, wrap=True)
    
    def arrow(x1, y1, x2, y2, text='', color='#333'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
        if text:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx+0.15, my, text, fontsize=7, color='#555', fontstyle='italic')
    
    # Input
    box(5.5, 9, 3, 0.7, 'Protein Sequence\n(AA + 3Di tokens)', '#333', 0.1, 10, True)
    arrow(7, 9, 7, 8.8)
    
    # SaProt backbone
    box(4.5, 7.8, 5, 0.8, 'SaProt Backbone (650M params, FROZEN)\n1280-dim structure-aware embedding', '#6C757D', 0.2, 9, True)
    arrow(7, 7.8, 5, 7.1, '1280-dim')
    arrow(7, 7.8, 9, 7.1, '1280-dim')
    
    # Aux features
    box(0.5, 7.2, 3.5, 0.6, 'Tm Aux Features (9-dim)\nOGT prior, len, AA comp, TM flag', '#E69F00', 0.2, 7)
    box(10, 7.2, 3.5, 0.6, 'OGT Aux Features (8-dim)\nlen, AA comp, TM flag', '#E69F00', 0.2, 7)
    
    # Bottleneck projections
    box(0.5, 6.2, 3.5, 0.6, 'Bottleneck: Linear(9→64)\n+ LayerNorm + GELU', '#00B4D8', 0.15, 7)
    box(10, 6.2, 3.5, 0.6, 'Bottleneck: Linear(8→64)\n+ LayerNorm + GELU', '#00B4D8', 0.15, 7)
    arrow(2.25, 7.2, 2.25, 6.8)
    arrow(11.75, 7.2, 11.75, 6.8)
    
    # Concat
    box(3, 5.3, 3, 0.6, 'Concat → 1344-dim\n(1280 + 64)', '#0077B6', 0.15, 8)
    box(8, 5.3, 3, 0.6, 'Concat → 1344-dim\n(1280 + 64)', '#0077B6', 0.15, 8)
    arrow(2.25, 6.2, 4.5, 5.9)
    arrow(5, 7.1, 4.5, 5.9)
    arrow(11.75, 6.2, 9.5, 5.9)
    arrow(9, 7.1, 9.5, 5.9)
    
    # FC1 - TRANSFER LAYER
    box(3, 4.2, 3, 0.7, 'FC1: 1344→512\n+ LayerNorm + GELU + Drop(0.3)', '#0077B6', 0.25, 8, True)
    box(8, 4.2, 3, 0.7, 'FC1: 1344→512\n+ LayerNorm + GELU + Drop(0.3)', '#0077B6', 0.25, 8, True)
    arrow(4.5, 5.3, 4.5, 4.9)
    arrow(9.5, 5.3, 9.5, 4.9)
    
    # Annotation: Transfer layer
    ax.annotate('REPRESENTATIONAL\nTRANSFER LAYER\n(512-dim)', xy=(3, 4.55), xytext=(0.2, 4.5),
                fontsize=8, color='#c0392b', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.2))
    ax.text(0.2, 3.9, 'Retains: hydrophobic packing,\nsalt bridges, interface burial,\n2° structure, charge distribution',
            fontsize=6.5, color='#666', fontstyle='italic')
    
    # FC2 + residual
    box(3, 3.1, 3, 0.7, 'FC2: 512→256 + Residual\n+ LayerNorm + GELU + Drop(0.2)', '#0077B6', 0.15, 8)
    box(8, 3.1, 3, 0.7, 'FC2: 512→256 + Residual\n+ LayerNorm + GELU + Drop(0.2)', '#0077B6', 0.15, 8)
    arrow(4.5, 4.2, 4.5, 3.8)
    arrow(9.5, 4.2, 9.5, 3.8)
    
    # Output heads
    box(3, 1.8, 3, 0.8, 'Tm Head: 256→2\nμ_Tm, σ_Tm\n(Heteroscedastic NLL)', '#2ecc71', 0.2, 8, True)
    box(8, 1.8, 3, 0.8, 'OGT Head: 256→2\nμ_OGT, σ_OGT\n(Focal Huber + NLL)', '#2ecc71', 0.2, 8, True)
    arrow(4.5, 3.1, 4.5, 2.6)
    arrow(9.5, 3.1, 9.5, 2.6)
    
    # Labels
    ax.text(4.5, 1.4, 'Tm PATHWAY (disjoint)', ha='center', fontsize=10, fontweight='bold', color='#0077B6')
    ax.text(9.5, 1.4, 'OGT PATHWAY (disjoint)', ha='center', fontsize=10, fontweight='bold', color='#0077B6')
    
    # Gradient isolation annotation
    ax.text(7, 3.5, 'cos θ = 0\n(zero gradient\ninterference)', ha='center', fontsize=8, 
            color='#c0392b', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#ffeaea', edgecolor='#c0392b', alpha=0.8))
    
    # Info bottleneck annotation at bottom
    ax.text(7, 0.7, 'INFORMATION BOTTLENECK: 1280-dim → 512-dim → 4-dim\n'
            'Hidden layers retain biological features; final output loses all except temperature.',
            ha='center', fontsize=8, fontstyle='italic', color='#555',
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='#ccc'))
    
    ax.set_title("StableProt V9: Disjoint Multi-Head Bottleneck Architecture", fontsize=13, fontweight='bold', pad=10)
    
    out_path = os.path.join(OUT, "architecture_boilerplate.png")
    fig.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")

# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("  GENERATING NAR FIGURES FOR STABLEPROT MANUSCRIPT")
    print("=" * 70)
    
    result = plot_calibration()
    if result:
        y_true, y_pred, y_conf = result
        plot_confidence_spread(y_true, y_pred, y_conf)
    
    plot_overfitting_subplot()
    plot_architecture_boilerplate()
    
    print("\n✓ All figures generated.")
