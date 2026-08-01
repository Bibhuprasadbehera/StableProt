#!/usr/bin/env python3
"""Render publication-quality Gradient Interference Histogram (Shared vs Disjoint Architecture)."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = "/home/bibhu/Documents/temstampto"
JSON_PATH = os.path.join(PROJECT_ROOT, "paper/writeup/plots/gradient_interference_histogram.json")
OUT_PATH_PAPER = os.path.join(PROJECT_ROOT, "paper/writeup/plots/gradient_interference_histogram.png")
OUT_PATH_PRES = os.path.join(PROJECT_ROOT, "presentation/plots/gradient_interference_histogram.png")

with open(JSON_PATH, "r") as f:
    data = json.load(f)

v7_overall = np.array(data["v7_overall_cosine_similarities"])
v7_thermo = np.array(data["v7_thermophilic_cosine_similarities"])

mean_overall = np.mean(v7_overall)
mean_thermo = np.mean(v7_thermo)

# Style setup
sns.set_theme(style="whitegrid", font="sans-serif")
fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=300)

# Plot KDE & Histograms for V7 Shared
sns.histplot(
    v7_overall, bins=25, kde=True, color='#0077B6', alpha=0.45,
    label=f'Shared Backbone (Overall, Mean $\cos\\theta = {mean_overall:.3f}$)', ax=ax
)
sns.histplot(
    v7_thermo, bins=20, kde=True, color='#D55E00', alpha=0.55,
    label=f'Shared Backbone (Thermophilic, Mean $\cos\\theta = {mean_thermo:.3f}$)', ax=ax
)

# Vertical line at 0.0 for V9 Disjoint
ax.axvline(x=0.0, color='#10B981', linestyle='--', linewidth=3.5, label='StableProt V9 Disjoint ($\cos\\theta \equiv 0.000$, Zero Interference)')

# Highlight negative gradient conflict zone
ax.axvspan(-0.08, 0.0, color='#ef4444', alpha=0.08, label='Destructive Gradient Interference ($\cos\\theta < 0$)')

ax.set_xlim(-0.08, 0.03)
ax.set_xlabel("Gradient Cosine Similarity ($\cos\\theta_{T_m, \mathrm{OGT}}$)", fontsize=11, fontweight='semibold', labelpad=8)
ax.set_ylabel("Batch Evaluation Frequency", fontsize=11, fontweight='semibold', labelpad=8)
ax.set_title("Multi-Task Gradient Interference: Shared vs. Disjoint Architecture", fontsize=13, fontweight='bold', pad=12)

ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, fontsize=9.5)
plt.tight_layout()

fig.savefig(OUT_PATH_PAPER, dpi=300, bbox_inches='tight')
fig.savefig(OUT_PATH_PRES, dpi=300, bbox_inches='tight')
plt.close(fig)

print("Rendered non-empty gradient_interference_histogram.png successfully!")
