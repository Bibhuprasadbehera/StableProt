#!/usr/bin/env python3
"""
Figure 1 (v5) — data pipeline and model architecture.

Rebuild of the v4 schematic. Changes:
  * data provenance and curation are shown explicitly (they were a dashed line before)
  * the 3Di dual-track tokenisation is drawn rather than named
  * the auxiliary bottlenecks are broken out, so the 9-vs-8 asymmetry is visible --
    the OGT prior enters the Tm path only, which is the point of the disjoint design
  * two-stage inference (OGT head feeds the Tm prior) is shown; it was absent
  * boxes are sized to their text, and the subsampling inset no longer overlaps

Two panels only: pipeline (A) and architecture (B).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import PALETTE, OUT, apply as apply_style  # noqa: E402

apply_style()


def box(ax, x0, y0, x1, y1, *, fc, ec=None, lw=1.0, ls="-", z=2, radius=1.2):
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=fc, edgecolor=ec or PALETTE["spine"],
            linewidth=lw, linestyle=ls, zorder=z,
        )
    )


def label(ax, x, y, text, *, size=7.5, weight="normal", color=None, style="normal",
          ha="center", va="center", z=4):
    ax.text(x, y, text, ha=ha, va=va, fontsize=size, fontweight=weight,
            color=color or PALETTE["spine"], fontstyle=style, zorder=z, linespacing=1.5)


def arrow(ax, xy_from, xy_to, *, color=None, lw=1.1, ls="-", z=3):
    ax.annotate(
        "", xy=xy_to, xytext=xy_from, zorder=z,
        arrowprops=dict(arrowstyle="-|>", color=color or PALETTE["muted"], linewidth=lw,
                        linestyle=ls, shrinkA=0, shrinkB=0),
    )


def blank_axes(fig, cell, title, letter):
    ax = fig.add_subplot(cell)
    ax.set_xlim(-2, 102)
    ax.set_ylim(0, 104)
    ax.axis("off")
    ax.text(-2, 103, letter, fontsize=14, fontweight="bold", va="top", ha="left",
            color=PALETTE["spine"], zorder=6)
    ax.text(3.5, 102.6, title, fontsize=11.5, fontweight="bold", va="top", ha="left",
            color=PALETTE["spine"], zorder=6)
    return ax


def stacked_bars(ax, x0, y0, w, h, series, categories, colors):
    """Small stacked bar chart drawn in data coordinates (avoids inset-axes collisions)."""
    n = len(categories)
    slot = w / n
    bw = slot * 0.46
    for i, cat in enumerate(categories):
        cx = x0 + slot * (i + 0.5)
        bottom = 0.0
        for val, col in zip(series[i], colors):
            ax.add_patch(
                mpatches.Rectangle((cx - bw / 2, y0 + h * bottom / 100.0), bw,
                                   h * val / 100.0, facecolor=col, edgecolor="white",
                                   linewidth=0.5, zorder=3)
            )
            bottom += val
        label(ax, cx, y0 - 2.6, cat, size=6.0, color=PALETTE["muted"])
    ax.plot([x0, x0], [y0, y0 + h], color=PALETTE["muted"], lw=0.7, zorder=3)
    for frac, txt in ((0.0, "0"), (0.5, "50"), (1.0, "100")):
        ax.plot([x0 - 0.9, x0], [y0 + h * frac] * 2, color=PALETTE["muted"], lw=0.7, zorder=3)
        label(ax, x0 - 1.8, y0 + h * frac, txt, size=5.6, ha="right", color=PALETTE["muted"])


# ═══════════════════════════════════════════════════════════════════════════
# Panel A — curation, decontamination, structure-aware representation
# ═══════════════════════════════════════════════════════════════════════════
def panel_a(fig, cell):
    ax = blank_axes(fig, cell, "Data curation and structure-aware representation", "A")

    # ── Row 1: sources → curation → splits ────────────────────────────────
    box(ax, 0, 74, 19, 96, fc=PALETTE["neutral"])
    label(ax, 9.5, 92.5, "Data sources", size=8, weight="bold")
    label(ax, 9.5, 84, "ProThermDB\nMeltome / FLIP\nBacDive + UniProt",
          size=7, color=PALETTE["muted"])
    arrow(ax, (19.8, 85), (24.7, 85))

    box(ax, 24.0, 74, 58.0, 96, fc=PALETTE["neutral"])
    label(ax, 41, 92.5, "Multi-stage curation", size=8, weight="bold")
    label(ax, 41, 83.5,
          "Dedup & Range Checks (43k → 29.3k $T_m$)\n"
          "Thermodynamic filter ($T_m \\geq$ OGT)\n"
          "Quality filter ($|\\mathrm{OGT\\,err}| \\leq 15^\\circ\\mathrm{C}$)\n"
          "Mesophilic subsampling (14% kept)",
          size=6.3, color=PALETTE["muted"])
    arrow(ax, (58.5, 85), (63.0, 85))

    box(ax, 63.8, 74, 80.5, 96, fc=PALETTE["OGT_wash"], ec=PALETTE["OGT"])
    label(ax, 72.1, 92.5, "Training set", size=8, weight="bold", color="#8A5424")
    label(ax, 72.1, 83.8, "28,739  $T_m$\n131,920  OGT\n(14% balanced)", size=7.2, color="#8A5424")

    box(ax, 85.5, 74, 101, 96, fc=PALETTE["boundary_wash"], ec=PALETTE["boundary"])
    label(ax, 93.2, 92.5, "Held-out evaluation", size=7.6, weight="bold",
          color=PALETTE["boundary"])
    label(ax, 93.2, 84, "ProThermDB\nFireProtDB\nBRENDA · BacDive",
          size=7, color=PALETTE["boundary"])

    ax.plot([83, 83], [72, 98], color=PALETTE["boundary"], lw=1.6, ls=(0, (4, 3)), zorder=3)
    label(ax, 101, 69,
          "Homology decontamination:  MMseqs2 < 30 % sequence identity ($T_m$)"
          "  ·  CD-HIT 40 % (OGT)",
          size=6.7, style="italic", color=PALETTE["boundary"], ha="right")

    # ── Row 2: representation chain ───────────────────────────────────────
    label(ax, 0, 60, "Structure-aware representation", size=8.5, weight="bold", ha="left")

    chain = [
        (0.0, 15.0, "Amino acid\nsequence", PALETTE["neutral"], PALETTE["spine"]),
        (17.0, 33.0, "ESMFold\nstructure\nprediction", PALETTE["neutral"], PALETTE["spine"]),
        (35.0, 51.0, "Foldseek\n3Di alphabet\n(20 states)",
         PALETTE["StableProt_pale"], PALETTE["StableProt"]),
        (53.0, 69.0, "Dual-track\ntokenisation",
         PALETTE["StableProt_pale"], PALETTE["StableProt"]),
        (71.0, 87.0, "SaProt 650M\nencoder",
         PALETTE["StableProt_pale"], PALETTE["StableProt"]),
    ]
    for x0, x1, txt, fc, tc in chain:
        box(ax, x0, 38, x1, 54, fc=fc)
        label(ax, (x0 + x1) / 2, 46, txt, size=7.2, color=tc)
    for x0, _, _, _, _ in chain[1:]:
        arrow(ax, (x0 - 1.8, 46), (x0 - 0.3, 46))

    box(ax, 89, 38, 101, 54, fc=PALETTE["StableProt"], ec=PALETTE["StableProt"])
    label(ax, 95, 46, "1280-d\nvector", size=7.2, weight="bold", color="white")
    arrow(ax, (87.3, 46), (88.4, 46))

    # dual-track detail, hung below the tokenisation box
    box(ax, 42, 6, 90, 26, fc="#FFFFFF", ec=PALETTE["StableProt"], ls=(0, (3, 2)), lw=0.9)
    label(ax, 66, 22, "one token per residue  =  amino acid  ⊕  3Di structural state",
          size=6.7, style="italic", color=PALETTE["StableProt"])
    for i, t in enumerate(["M#d", "K#v", "V#p", "L#a", "E#d", "G#q", "I#v", "R#p"]):
        cx = 48.5 + i * 5.0
        box(ax, cx - 2.1, 10.5, cx + 2.1, 17.0, fc=PALETTE["StableProt_pale"],
            ec=PALETTE["StableProt"], lw=0.7, radius=0.6)
        label(ax, cx, 13.75, t, size=6.2, color=PALETTE["StableProt"])
    arrow(ax, (61, 37.4), (61, 26.8), color=PALETTE["StableProt"], lw=0.9, ls=":")

    # mesophilic subsampling, in the free area left of the token strip
    label(ax, 3, 29, "Mesophilic subsampling of the OGT set",
          size=6.9, weight="bold", ha="left")
    stacked_bars(ax, 8, 10, 22, 16,
                 series=[[87, 13], [62, 38]], categories=["Raw", "Subsampled"],
                 colors=[PALETTE["meso"], PALETTE["thermo"]])
    label(ax, 1.8, 18, "% of set", size=5.8, color=PALETTE["muted"], ha="center")
    for i, (col, txt) in enumerate(((PALETTE["meso"], "Mesophile"),
                                    (PALETTE["thermo"], "≥ Thermophile"))):
        ax.add_patch(mpatches.Rectangle((5 + i * 14, 1.4), 2.0, 2.0, facecolor=col,
                                        edgecolor="none", zorder=3))
        label(ax, 7.8 + i * 14, 2.4, txt, size=5.8, ha="left", color=PALETTE["muted"])


# ═══════════════════════════════════════════════════════════════════════════
# Panel B — disjoint architecture and two-stage inference
# ═══════════════════════════════════════════════════════════════════════════
def panel_b(fig, cell):
    ax = blank_axes(fig, cell, "Disjoint multi-head architecture", "B")
    NAVY, AMBER = PALETTE["StableProt"], PALETTE["OGT"]

    box(ax, 34, 86, 66, 95, fc=NAVY, ec=NAVY)
    label(ax, 50, 90.5, "SaProt embedding  ·  1280-d", size=8, weight="bold", color="white")
    arrow(ax, (43, 85.4), (26, 79.6), color=NAVY, lw=1.2)
    arrow(ax, (57, 85.4), (74, 79.6), color=AMBER, lw=1.2)

    # ── auxiliary features: the asymmetry ─────────────────────────────────
    box(ax, 1, 63, 47, 79, fc=PALETTE["StableProt_wash"], ec=NAVY)
    label(ax, 24, 75.5, "$T_m$ auxiliary features  (9)", size=8, weight="bold", color=NAVY)
    label(ax, 24, 68.5,
          "predicted OGT prior  ·  TM-helix flag\nsequence length  ·  6 amino-acid ratios",
          size=7, color=NAVY)

    box(ax, 53, 63, 99, 79, fc=PALETTE["OGT_wash"], ec=AMBER)
    label(ax, 76, 75.5, "OGT auxiliary features  (8)", size=8, weight="bold", color=AMBER)
    label(ax, 76, 68.5,
          "TM-helix flag\nsequence length  ·  6 amino-acid ratios", size=7, color=AMBER)

    box(ax, 29, 54.5, 71, 60.5, fc="#FFFFFF", ec=PALETTE["muted"], ls=(0, (3, 2)),
        lw=0.9, radius=0.8)
    label(ax, 50, 57.5, "the OGT prior enters the $T_m$ path only", size=7,
          style="italic", color=PALETTE["muted"])

    # ── bottleneck projections ────────────────────────────────────────────
    for x0, x1, txt, col, wash in ((7, 41, "Linear  9 → 64", NAVY, PALETTE["StableProt_pale"]),
                                   (59, 93, "Linear  8 → 64", AMBER, PALETTE["OGT_pale"])):
        cx = (x0 + x1) / 2
        arrow(ax, (cx, 62.4), (cx, 50.6), color=col, lw=1.2)
        box(ax, x0, 43.5, x1, 50, fc=wash, ec=col)
        label(ax, cx, 46.75, txt, size=7.5, weight="bold", color=col)
        label(ax, cx + 1.8, 39.8, "concatenate → 1344-d", size=6.8, style="italic",
              color=PALETTE["muted"], ha="left")
        arrow(ax, (cx, 42.9), (cx, 36.6), color=col, lw=1.2)

    # ── prediction heads ──────────────────────────────────────────────────
    box(ax, 1, 13, 47, 36, fc=PALETTE["StableProt_wash"], ec=NAVY, lw=1.4)
    label(ax, 24, 32.5, "$T_m$ head", size=9, weight="bold", color=NAVY)
    label(ax, 24, 25.5, "1344 → 512 → 256 → 2", size=7.5, color=NAVY)
    label(ax, 24, 20.5, "$\\mu$   and   $\\sigma^2 = \\mathrm{Softplus}(v) + 10^{-4}$",
          size=7.5, color=NAVY)
    label(ax, 24, 15.8, "Gaussian negative log-likelihood", size=7, style="italic", color=NAVY)

    box(ax, 53, 13, 99, 36, fc=PALETTE["OGT_wash"], ec=AMBER, lw=1.4)
    label(ax, 76, 32.5, "OGT head", size=9, weight="bold", color=AMBER)
    label(ax, 76, 25.5, "1344 → 512 → 256 → 2", size=7.5, color=AMBER)
    label(ax, 76, 20.5, "$\\mu$   and   $\\sigma^2$", size=7.5, color=AMBER)
    label(ax, 76, 15.8, "focal Huber  +  detached-mean NLL", size=7, style="italic", color=AMBER)

    # ── inference / training notes ────────────────────────────────────────
    box(ax, 1, 0.5, 99, 10, fc=PALETTE["neutral"], ec=PALETTE["muted"], lw=0.9)
    label(ax, 50, 6.9,
          "Two-stage inference:   OGT head → $\\hat{y}_{\\mathrm{OGT}}$ → "
          "OGT prior in the $T_m$ auxiliary vector → $T_m$ head → $\\mu \\pm \\sigma$",
          size=7.0, color=PALETTE["spine"])
    label(ax, 50, 3.1,
          "During training the OGT prior is perturbed with scheduled Gaussian noise "
          "($\\sigma = 2.0\\ ^\\circ$C).   Reported predictions average 5 seeds.",
          size=6.5, style="italic", color=PALETTE["muted"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=OUT)
    args = p.parse_args()
    dest = Path(args.out_dir)
    fig = plt.figure(figsize=(7.4, 8.8))
    gs = GridSpec(2, 1, height_ratios=[1.0, 1.0], hspace=0.10,
                  left=0.045, right=0.985, top=0.978, bottom=0.015)
    panel_a(fig, gs[0])
    panel_b(fig, gs[1])
    dest.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(dest / f"fig1_architecture_pipeline.{ext}", dpi=300)
    plt.close(fig)
    print(f"  wrote {dest/'fig1_architecture_pipeline.png'} (+ .svg)")


if __name__ == "__main__":
    main()
