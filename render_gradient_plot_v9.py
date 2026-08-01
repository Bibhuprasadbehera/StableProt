#!/usr/bin/env python3
"""Render original white-background Gradient Interference Histogram with updated StableProt V9 label text."""

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

shared_overall = np.array(data["v7_overall_cosine_similarities"])
shared_thermo = np.array(data["v7_thermophilic_cosine_similarities"])

mean_overall = np.mean(shared_overall)
mean_thermo = np.mean(shared_thermo)

# Clean White Background Style (Original Paper Palette)
sns.set_theme(style="white", font="sans-serif")
fig, ax = plt.subplots(figsize=(9.0, 5.0), dpi=300)

fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Plot Shared Backbone distributions with original soft blue and soft red colors
sns.histplot(
    shared_overall, bins=30, kde=True, color='#b0c4de', alpha=0.8,
    label=f'Shared Backbone Overall Batches (Mean cos $\\theta = {mean_overall:.2f}$)', ax=ax,
    edgecolor='black', linewidth=0.6
)
sns.histplot(
    shared_thermo, bins=25, kde=True, color='#f08080', alpha=0.7,
    label=f'Shared Backbone Thermophilic vs OGT Background (Mean cos $\\theta = {mean_thermo:.2f}$)', ax=ax,
    edgecolor='black', linewidth=0.6
)

# Vertical dashed line for StableProt V9 Disjoint Multi-Head Architecture
ax.axvline(x=0.0, color='#10b981', linestyle='--', linewidth=3.0, label='StableProt V9 Disjoint Multi-Head ($\cos\\theta \equiv 0.00$)')

ax.set_xlim(-0.07, 0.005)
ax.set_xlabel("Gradient Cosine Similarity ($\cos\\theta_{T_m, \mathrm{OGT}}$)\nNegative values (< 0) indicate conflicting parameter updates across tasks", fontsize=11, color='black', labelpad=8)
ax.set_ylabel("Batch Evaluation Frequency", fontsize=11, color='black', labelpad=8)
ax.set_title("Multi-Task Gradient Interference: Shared vs. Decoupled Architecture", fontsize=13, color='black', pad=10)

ax.tick_params(colors='black', labelsize=10)
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(1.0)

# Legend
leg = ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='black', fontsize=9.5)
for text in leg.get_texts():
    text.set_color('black')

plt.tight_layout()

fig.savefig(OUT_PATH_PAPER, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(OUT_PATH_PRES, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)

print("Rendered white-background StableProt V9 gradient interference plot successfully!")
