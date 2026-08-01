#!/usr/bin/env python3
"""Render clean high-impact Matplotlib barplot for Emergent Transfer Benchmarks."""

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
fig.patch.set_facecolor('#0B0F19')
ax.set_facecolor('#0B0F19')

tasks = ['Human PPI\nAccuracy', 'LiveProteinBench\nTemp Correlation (r)', 'eSOL Solubility\nLinear Probe (R²)', 'DeepLoc Subcellular\nLocalization Acc']
scores = [88.33, 54.12, 35.40, 85.00]
colors = ['#34D399', '#38BDF8', '#C084FC', '#F59E0B']

bars = ax.bar(tasks, scores, color=colors, width=0.5, edgecolor='#1E293B', linewidth=1.5)

for bar, score in zip(bars, scores):
    height = bar.get_height()
    if score > 1.0:
        val_str = f"{score:.1f}%"
    else:
        val_str = f"{score:.3f}"
    if 'Correlation' in tasks[bars.index(bar)]:
        val_str = f"r = 0.541"
    elif 'Solubility' in tasks[bars.index(bar)]:
        val_str = f"R² = 0.354"
        
    ax.text(bar.get_x() + bar.get_width()/2., height + 2.0, val_str,
            ha='center', va='bottom', color='#FFFFFF', fontsize=10, fontweight='bold')

ax.set_ylabel('Performance Score', color='#94A3B8', fontsize=11, fontweight='bold')
ax.set_title('StableProt V9 Emergent Representation Transfer Scores', color='#FFFFFF', fontsize=13, fontweight='bold', pad=15)

ax.tick_params(colors='#94A3B8', labelsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#1E293B')
ax.spines['bottom'].set_color('#1E293B')
ax.set_ylim(0, 100)
ax.grid(axis='y', linestyle='--', alpha=0.2, color='#94A3B8')

plt.tight_layout()
plt.savefig('/home/bibhu/Documents/temstampto/presentation/plots/emergent_transfer_barplot.png', 
            bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
plt.savefig('/home/bibhu/Documents/temstampto/paper/writeup/plots/emergent_transfer_barplot.png', 
            bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
print("Successfully rendered emergent_transfer_barplot.png!")
