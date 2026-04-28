#!/usr/bin/env python3
"""
Plot temperature distribution across all TemStaPro dataset FASTA files.
Extracts OGT (Optimal Growth Temperature) from FASTA headers (format: >taxid|uniprot_id|temperature)
and generates comprehensive distribution plots.
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

DATASET_DIR = "/home/bibhu/Documents/temstampto/dataset"
OUTPUT_DIR = "/home/bibhu/Documents/temstampto/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Color palette ──
COLORS = {
    'training':   '#6366f1',   # indigo
    'validation': '#f59e0b',   # amber
    'testing':    '#10b981',   # emerald
    'minor_cv':   '#ef4444',   # red
    'minor_test': '#8b5cf6',   # violet
}

def extract_temperatures(fasta_path):
    """Extract temperatures from FASTA headers."""
    temps = []
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                parts = line.strip().lstrip('>').split('|')
                if len(parts) >= 3:
                    try:
                        temp = float(parts[2])
                        temps.append(temp)
                    except ValueError:
                        pass
    return temps

# ── Extract temperatures from all main split files ──
print("Extracting temperatures from dataset files...")

files_info = {
    'Training (Major-Imbal)':   ('TemStaPro-Major-30-imbal-training.fasta', 'training'),
    'Validation (Major-Imbal)': ('TemStaPro-Major-30-imbal-validation.fasta', 'validation'),
    'Testing (Major-Imbal)':    ('TemStaPro-Major-30-imbal-testing.fasta', 'testing'),
    'Cross-Val (Minor)':        ('TemStaPro-Minor-30-cross-validation.fasta', 'minor_cv'),
    'Testing (Minor)':          ('TemStaPro-Minor-30-testing.fasta', 'minor_test'),
}

all_data = {}
for label, (fname, color_key) in files_info.items():
    fpath = os.path.join(DATASET_DIR, fname)
    if os.path.exists(fpath):
        temps = extract_temperatures(fpath)
        all_data[label] = {'temps': temps, 'color': COLORS[color_key]}
        print(f"  {label}: {len(temps):,} sequences, temp range [{min(temps):.0f}, {max(temps):.0f}]°C")

# Also extract from the balanced testing samples
balanced_files = sorted(glob.glob(os.path.join(DATASET_DIR, "TemStaPro-Major-30-bal*-testing-sample2k.fasta")))
balanced_data = {}
for fpath in balanced_files:
    fname = os.path.basename(fpath)
    # e.g. TemStaPro-Major-30-bal40-testing-sample2k.fasta -> bal40
    bal_label = fname.split('-')[3]  # bal40, bal45, ...
    temps = extract_temperatures(fpath)
    balanced_data[bal_label] = temps
    print(f"  Balanced {bal_label}: {len(temps):,} sequences")

# ── Combine all temperatures for global view ──
all_temps = []
for info in all_data.values():
    all_temps.extend(info['temps'])

print(f"\nTotal sequences across all splits: {len(all_temps):,}")
print(f"Global temperature range: [{min(all_temps):.0f}, {max(all_temps):.0f}]°C")
print(f"Mean: {np.mean(all_temps):.1f}°C, Median: {np.median(all_temps):.1f}°C, Std: {np.std(all_temps):.1f}°C")

# ═══════════════════════════════════════════════════
# PLOT 1: Combined histogram of all splits (stacked)
# ═══════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 7))

bins = np.arange(0, max(all_temps) + 5, 2)  # 2°C bins

for label, info in all_data.items():
    ax.hist(info['temps'], bins=bins, alpha=0.55, label=f"{label} (n={len(info['temps']):,})",
            color=info['color'], edgecolor='white', linewidth=0.5)

ax.set_xlabel('Optimal Growth Temperature (°C)', fontsize=13, fontweight='bold')
ax.set_ylabel('Number of Sequences', fontsize=13, fontweight='bold')
ax.set_title('Temperature Distribution Across All Dataset Splits', fontsize=16, fontweight='bold', pad=15)
ax.legend(fontsize=10, framealpha=0.9, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add statistics annotation
stats_text = (f"Total: {len(all_temps):,} sequences\n"
              f"Mean: {np.mean(all_temps):.1f}°C\n"
              f"Median: {np.median(all_temps):.1f}°C\n"
              f"Std: {np.std(all_temps):.1f}°C")
ax.text(0.02, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_combined_temp_distribution.png'), dpi=200, bbox_inches='tight')
plt.close()
print("\nSaved: 01_combined_temp_distribution.png")


# ═══════════════════════════════════════════════════
# PLOT 2: Individual subplots per split
# ═══════════════════════════════════════════════════
fig, axes = plt.subplots(len(all_data), 1, figsize=(14, 3.5 * len(all_data)), sharex=True)

for idx, (label, info) in enumerate(all_data.items()):
    ax = axes[idx]
    temps = info['temps']
    ax.hist(temps, bins=bins, alpha=0.75, color=info['color'], edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'{label}  (n={len(temps):,}, mean={np.mean(temps):.1f}°C, median={np.median(temps):.1f}°C)',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add vertical lines for mean and median
    ax.axvline(np.mean(temps), color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean={np.mean(temps):.1f}°C')
    ax.axvline(np.median(temps), color='navy', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Median={np.median(temps):.1f}°C')
    ax.legend(fontsize=9, loc='upper right')

axes[-1].set_xlabel('Optimal Growth Temperature (°C)', fontsize=13, fontweight='bold')
fig.suptitle('Temperature Distribution Per Dataset Split', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_per_split_temp_distribution.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 02_per_split_temp_distribution.png")


# ═══════════════════════════════════════════════════
# PLOT 3: CDF (Cumulative Distribution Function)
# ═══════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))

for label, info in all_data.items():
    sorted_temps = np.sort(info['temps'])
    cdf = np.arange(1, len(sorted_temps) + 1) / len(sorted_temps)
    ax.plot(sorted_temps, cdf, linewidth=2, label=f"{label} (n={len(info['temps']):,})",
            color=info['color'], alpha=0.85)

# Add threshold lines
thresholds = [40, 45, 50, 55, 60, 65]
for t in thresholds:
    ax.axvline(t, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.text(t, 1.02, f'{t}°C', ha='center', fontsize=8, color='gray', transform=ax.get_xaxis_transform())

ax.set_xlabel('Optimal Growth Temperature (°C)', fontsize=13, fontweight='bold')
ax.set_ylabel('Cumulative Proportion', fontsize=13, fontweight='bold')
ax.set_title('Cumulative Distribution of Temperatures', fontsize=16, fontweight='bold', pad=15)
ax.legend(fontsize=10, framealpha=0.9, loc='lower right')
ax.grid(alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_cdf_temp_distribution.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 03_cdf_temp_distribution.png")


# ═══════════════════════════════════════════════════
# PLOT 4: Box + Violin plot comparison across splits
# ═══════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

labels = list(all_data.keys())
data_list = [all_data[l]['temps'] for l in labels]
colors_list = [all_data[l]['color'] for l in labels]

# Violin plot
vparts = ax1.violinplot(data_list, positions=range(len(labels)), showmeans=True, showmedians=True)
for i, pc in enumerate(vparts['bodies']):
    pc.set_facecolor(colors_list[i])
    pc.set_alpha(0.6)
vparts['cmeans'].set_color('red')
vparts['cmedians'].set_color('navy')

ax1.set_xticks(range(len(labels)))
ax1.set_xticklabels([l.replace(' (', '\n(') for l in labels], fontsize=9)
ax1.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
ax1.set_title('Violin Plot', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Box plot
bp = ax2.boxplot(data_list, patch_artist=True, notch=True, widths=0.6)
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(colors_list[i])
    patch.set_alpha(0.6)
for median in bp['medians']:
    median.set_color('navy')
    median.set_linewidth(2)

ax2.set_xticks(range(1, len(labels) + 1))
ax2.set_xticklabels([l.replace(' (', '\n(') for l in labels], fontsize=9)
ax2.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
ax2.set_title('Box Plot', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fig.suptitle('Temperature Distribution Comparison Across Splits', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_violin_box_comparison.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 04_violin_box_comparison.png")


# ═══════════════════════════════════════════════════
# PLOT 5: Thermophile class balance per threshold
# ═══════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))

thresholds = [40, 45, 50, 55, 60, 65, 70, 75, 80]
x = np.arange(len(thresholds))
width = 0.15

for i, (label, info) in enumerate(all_data.items()):
    temps = np.array(info['temps'])
    thermo_pct = [100 * np.sum(temps >= t) / len(temps) for t in thresholds]
    bars = ax.bar(x + i * width, thermo_pct, width, label=label, color=info['color'], alpha=0.8, edgecolor='white')

    # Add value labels on bars
    for bar, pct in zip(bars, thermo_pct):
        if pct > 3:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=6.5, fontweight='bold')

ax.set_xlabel('Temperature Threshold (°C)', fontsize=13, fontweight='bold')
ax.set_ylabel('% Sequences ≥ Threshold', fontsize=13, fontweight='bold')
ax.set_title('Thermophile Proportion at Different Temperature Thresholds', fontsize=16, fontweight='bold', pad=15)
ax.set_xticks(x + width * (len(all_data) - 1) / 2)
ax.set_xticklabels([f'{t}°C' for t in thresholds], fontsize=11)
ax.legend(fontsize=9, framealpha=0.9, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05_thermophile_balance.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 05_thermophile_balance.png")


# ═══════════════════════════════════════════════════
# PLOT 6: Balanced vs Imbalanced test sets comparison
# ═══════════════════════════════════════════════════
if balanced_data:
    n_bal = len(balanced_data)
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    axes = axes.flatten()

    bal_colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_bal))

    for idx, (bal_label, temps) in enumerate(sorted(balanced_data.items())):
        if idx < len(axes):
            ax = axes[idx]
            ax.hist(temps, bins=bins, alpha=0.75, color=bal_colors[idx], edgecolor='white', linewidth=0.5)
            threshold = int(bal_label.replace('bal', ''))
            thermo_count = sum(1 for t in temps if t >= threshold)
            meso_count = len(temps) - thermo_count
            ax.set_title(f'{bal_label} (n={len(temps):,})\nThermo(≥{threshold}°C)={thermo_count} | Meso={meso_count}',
                         fontsize=10, fontweight='bold')
            ax.axvline(threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylabel('Count', fontsize=9)

    # Hide unused axes
    for idx in range(n_bal, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Balanced Testing Samples – Temperature Distribution\n(Red line = classification threshold)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '06_balanced_test_samples.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: 06_balanced_test_samples.png")


# ═══════════════════════════════════════════════════
# PLOT 7: Summary statistics table as a figure
# ═══════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 4))
ax.axis('off')

table_data = []
for label, info in all_data.items():
    temps = np.array(info['temps'])
    table_data.append([
        label,
        f"{len(temps):,}",
        f"{np.min(temps):.0f}°C",
        f"{np.max(temps):.0f}°C",
        f"{np.mean(temps):.1f}°C",
        f"{np.median(temps):.1f}°C",
        f"{np.std(temps):.1f}°C",
        f"{np.percentile(temps, 25):.1f}°C",
        f"{np.percentile(temps, 75):.1f}°C",
    ])

col_labels = ['Split', 'Count', 'Min', 'Max', 'Mean', 'Median', 'Std', 'Q1', 'Q3']
table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.8)

# Style header
for j in range(len(col_labels)):
    table[0, j].set_facecolor('#4f46e5')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Alternate row colors
for i in range(1, len(table_data) + 1):
    color = '#f0f0ff' if i % 2 == 0 else 'white'
    for j in range(len(col_labels)):
        table[i, j].set_facecolor(color)

ax.set_title('Dataset Summary Statistics', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '07_summary_statistics.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 07_summary_statistics.png")


print(f"\n✅ All plots saved to: {OUTPUT_DIR}")
print("Files generated:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.endswith('.png'):
        print(f"  📊 {f}")
