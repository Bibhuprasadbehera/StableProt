#!/usr/bin/env python3
"""Render clean, high-impact Matplotlib diagram for Slide 1 showing Amino Acid Sequence ONLY input."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(10, 3.5), dpi=300)
fig.patch.set_facecolor('#0B0F19')
ax.set_facecolor('#0B0F19')
ax.axis('off')

# Box 1: Input (FASTA Sequence ONLY)
box1 = patches.FancyBboxPatch((0.5, 0.8), 2.2, 1.8, boxstyle="round,pad=0.2,rounding_size=0.15",
                             facecolor='#0F172A', edgecolor='#38BDF8', linewidth=2)
ax.add_patch(box1)
ax.text(1.6, 2.1, "1. USER INPUT", color='#38BDF8', fontsize=10, fontweight='bold', ha='center')
ax.text(1.6, 1.6, "Amino Acid Sequence ONLY", color='#FFFFFF', fontsize=11, fontweight='bold', ha='center')
ax.text(1.6, 1.1, "(Standard Single-Letter FASTA)\nNo PDB upload required!", color='#94A3B8', fontsize=8, ha='center', style='italic')

# Arrow 1
ax.annotate("", xy=(3.3, 1.7), xytext=(2.9, 1.7),
            arrowprops=dict(arrowstyle="->", color="#38BDF8", lw=2.5))

# Box 2: Internal Structure Tokenization
box2 = patches.FancyBboxPatch((3.5, 0.8), 2.4, 1.8, boxstyle="round,pad=0.2,rounding_size=0.15",
                             facecolor='#0F172A', edgecolor='#A855F7', linewidth=2)
ax.add_patch(box2)
ax.text(4.7, 2.1, "2. INTERNAL INFERENCE", color='#C084FC', fontsize=10, fontweight='bold', ha='center')
ax.text(4.7, 1.6, "Automated 3Di Encoding", color='#FFFFFF', fontsize=11, fontweight='bold', ha='center')
ax.text(4.7, 1.1, "SaProt Foldseek 3Di Tokens\ngenerated automatically", color='#94A3B8', fontsize=8, ha='center')

# Arrow 2
ax.annotate("", xy=(6.5, 1.7), xytext=(6.1, 1.7),
            arrowprops=dict(arrowstyle="->", color="#C084FC", lw=2.5))

# Box 3: Predictions Output
box3 = patches.FancyBboxPatch((6.7, 0.8), 2.6, 1.8, boxstyle="round,pad=0.2,rounding_size=0.15",
                             facecolor='#0F172A', edgecolor='#34D399', linewidth=2)
ax.add_patch(box3)
ax.text(8.0, 2.1, "3. DUAL PREDICTIONS", color='#34D399', fontsize=10, fontweight='bold', ha='center')
ax.text(8.0, 1.6, "T_m & OGT ± Confidence", color='#FFFFFF', fontsize=11, fontweight='bold', ha='center')
ax.text(8.0, 1.1, "Melting Temp: T_m ± σ_Tm\nGrowth Temp: OGT ± σ_OGT", color='#34D399', fontsize=8.5, fontweight='bold', ha='center')

ax.set_xlim(0, 9.8)
ax.set_ylim(0.4, 2.6)

plt.tight_layout()
plt.savefig('/home/bibhu/Documents/temstampto/presentation/plots/slide1_simple_pipeline.png', 
            bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
plt.savefig('/home/bibhu/Documents/temstampto/paper/writeup/plots/slide1_simple_pipeline.png', 
            bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
print("Successfully rendered custom slide1_simple_pipeline.png diagram!")
