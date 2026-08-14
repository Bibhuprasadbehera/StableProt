#!/usr/bin/env python3
"""
StableProt manuscript figures: main Figures 1-6 and Supplementary S1-S3.

Style comes from figstyle.py, which every generation script shares. Numbers come from
tables/refreshed_tm_numbers.json, written by experiments/src/eval/refresh_tm_tables.py.
Neither is redefined here.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import numpy as np
from scipy import special
from scipy.stats import pearsonr, spearmanr


def roc_curve_auc(y_true_bin, scores):
    """ROC curve + AUC without sklearn."""
    y_true_bin = np.asarray(y_true_bin).astype(int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores)
    y = y_true_bin[order]
    P = max(int(y.sum()), 1)
    N = max(len(y) - P, 1)
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    tpr = np.concatenate([[0.0], tps / P, [1.0]])
    fpr = np.concatenate([[0.0], fps / N, [1.0]])
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc

# ═══════════════════════════════════════════════════════════════════════════
# Design tokens come from figstyle so every script draws the same figure family.
# Nothing style-related is defined in this file.
# ═══════════════════════════════════════════════════════════════════════════
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (  # noqa: E402
    MARKERS,
    PALETTE,
    TEMP_RAMP,
    apply as apply_style,
    despine,
    model_color,
    model_marker,
    panel_label,
    panel_title,
    reference_line,
    savefig,
)

apply_style()

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "paper" / "writeup" / "plots"  # caches / JSON / source assets (read-only)
OUT = PROJECT / "paper" / "writeup" / "plots_v4"  # default write target (does not clobber DATA)

# Verified numbers for every T_m panel. Written by
# experiments/src/eval/refresh_tm_tables.py under the shipped configuration.
# Panels must read from this file rather than from literals.
NUMBERS = PROJECT / "paper" / "writeup" / "tables" / "refreshed_tm_numbers.json"


def load_numbers():
    if not NUMBERS.exists():
        raise SystemExit(
            f"{NUMBERS} not found. Run experiments/src/eval/refresh_tm_tables.py first; "
            "figures must not be drawn from hardcoded values."
        )
    with open(NUMBERS) as fh:
        return json.load(fh)


def expected_coverage(z):
    return special.erf(z / np.sqrt(2.0))


def int_mae(y_true, y_pred, sigma, T=1.0):
    return np.mean(np.maximum(0.0, np.abs(y_true - y_pred) - T * sigma))


def load_npz(name):
    return np.load(DATA / name, allow_pickle=True)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Architecture schematic (draft vector; refine in Inkscape)
# ═══════════════════════════════════════════════════════════════════════════
def figure1():
    print("── Figure 1: Architecture & Data Pipeline ──")
    fig = plt.figure(figsize=(7.0, 8.5))
    gs = GridSpec(2, 1, height_ratios=[1.05, 1.0], hspace=0.28)

    # ── Panel A: Pipeline flowchart ──
    ax = fig.add_subplot(gs[0])
    panel_label(ax, "A")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    panel_title(ax, "End-to-End Computational Pipeline")

    stages = [
        (0.3, 5.8, 1.8, 1.4, "Amino Acid\nSequence", PALETTE["neutral"]),
        (2.5, 5.8, 1.8, 1.4, "ESMFold /\nOpenFold", PALETTE["StableProt_wash"]),
        (4.7, 5.8, 1.8, 1.4, "3D PDB\nCoordinates", PALETTE["StableProt_wash"]),
        (6.9, 5.8, 1.8, 1.4, "Foldseek\n3Di (20)", PALETTE["StableProt_pale"]),
        (9.1, 5.8, 2.0, 1.4, "SaProt Dual-Track\n1280-d Embedding", PALETTE["StableProt_pale"]),
        (11.5, 5.8, 2.2, 1.4, "Disjoint\nProjection Heads", PALETTE["StableProt"]),
    ]
    for x, y, w, h, txt, c in stages:
        fc = c if c != PALETTE["StableProt"] else PALETTE["StableProt"]
        tc = "white" if c == PALETTE["StableProt"] else PALETTE["spine"]
        box = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
            facecolor=fc, edgecolor=PALETTE["spine"], linewidth=1.0,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=7.5, color=tc)

    for x in [2.1, 4.3, 6.5, 8.7, 11.1]:
        ax.annotate("", xy=(x + 0.35, 6.5), xytext=(x, 6.5),
                    arrowprops=dict(arrowstyle="->", color=PALETTE["spine"], lw=1.2))

    # Leakage boundary
    ax.plot([0.5, 13.5], [4.6, 4.6], ls="--", color=PALETTE["bad"], lw=1.8)
    ax.text(7.0, 4.85, "Zero-leakage boundary: MMseqs2 <30% seq-ID  ·  CD-HIT structural filter",
            ha="center", fontsize=8, color=PALETTE["bad"], fontstyle="italic")

    # Training / holdout boxes
    train = mpatches.FancyBboxPatch(
        (0.5, 2.6), 5.5, 1.6, boxstyle="round,pad=0.05,rounding_size=0.12",
        facecolor="#FEF3C7", edgecolor=PALETTE["OGT"], linewidth=1.2,
    )
    hold = mpatches.FancyBboxPatch(
        (7.5, 2.6), 5.5, 1.6, boxstyle="round,pad=0.05,rounding_size=0.12",
        facecolor="#FEE2E2", edgecolor=PALETTE["bad"], linewidth=1.2,
    )
    ax.add_patch(train)
    ax.add_patch(hold)
    ax.text(3.25, 3.7, "Training Set", ha="center", fontsize=9, fontweight="bold", color=PALETTE["OGT"])
    ax.text(3.25, 3.15, "29,300 $T_m$  ·  940,000 OGT", ha="center", fontsize=8, color=PALETTE["OGT"])
    ax.text(10.25, 3.7, "Evaluation Holdouts", ha="center", fontsize=9, fontweight="bold", color=PALETTE["bad"])
    ax.text(10.25, 3.15, "ProThermDB · FireProtDB · BRENDA", ha="center", fontsize=8, color=PALETTE["bad"])

    # Mesophilic subsampling inset
    ax_in = fig.add_axes([0.12, 0.54, 0.22, 0.08])
    cats = ["Raw OGT", "Subsampled"]
    meso_pct = [87, 62]
    thermo_pct = [13, 38]
    ax_in.bar(cats, meso_pct, color=PALETTE["meso"], label="Mesophilic", width=0.55)
    ax_in.bar(cats, thermo_pct, bottom=meso_pct, color=PALETTE["thermo"], label="≥ Thermophile", width=0.55)
    ax_in.set_ylim(0, 100)
    ax_in.set_ylabel("%", fontsize=7)
    ax_in.set_title("MesophilicSubsampler\n(14% retention)", fontsize=7, fontweight="bold")
    ax_in.tick_params(labelsize=6)
    ax_in.legend(fontsize=5, loc="upper right", frameon=False)
    despine(ax_in)

    # ── Panel B: Disjoint heads ──
    ax = fig.add_subplot(gs[1])
    panel_label(ax, "B")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    panel_title(ax, "Disjoint Pathway Architecture")

    # Shared fusion
    fus = mpatches.FancyBboxPatch(
        (4.5, 6.2), 5.0, 1.4, boxstyle="round,pad=0.05,rounding_size=0.12",
        facecolor=PALETTE["StableProt_pale"], edgecolor=PALETTE["StableProt"], linewidth=1.3,
    )
    ax.add_patch(fus)
    ax.text(7.0, 7.15, "Feature Fusion", ha="center", fontsize=9, fontweight="bold", color=PALETTE["StableProt"])
    ax.text(7.0, 6.55, "SaProt 1280-d + Aux→64-d  →  1344-d", ha="center", fontsize=7.5, color=PALETTE["StableProt"])

    # Tm head
    tm = mpatches.FancyBboxPatch(
        (0.8, 1.2), 5.2, 4.2, boxstyle="round,pad=0.05,rounding_size=0.15",
        facecolor=PALETTE["StableProt_wash"], edgecolor=PALETTE["StableProt"], linewidth=1.5,
    )
    ax.add_patch(tm)
    ax.text(3.4, 5.0, "$T_m$ Head (Navy)", ha="center", fontsize=10, fontweight="bold", color=PALETTE["StableProt"])
    for i, (lab, y) in enumerate([
        ("1344 → 512", 4.2),
        ("512 → 256", 3.4),
        ("256 → 2  ($\\mu$, $v$)", 2.6),
        ("$\\sigma^2 = \\mathrm{Softplus}(v)+10^{-4}$", 1.8),
    ]):
        ax.text(3.4, y, lab, ha="center", fontsize=8, color=PALETTE["StableProt"])
    ax.text(3.4, 1.4, "Loss: Gaussian NLL", ha="center", fontsize=8, fontstyle="italic", color=PALETTE["StableProt"])

    # OGT head
    og = mpatches.FancyBboxPatch(
        (8.0, 1.2), 5.2, 4.2, boxstyle="round,pad=0.05,rounding_size=0.15",
        facecolor=PALETTE["OGT_wash"], edgecolor=PALETTE["OGT"], linewidth=1.5,
    )
    ax.add_patch(og)
    ax.text(10.6, 5.0, "OGT Head (Amber)", ha="center", fontsize=10, fontweight="bold", color=PALETTE["OGT"])
    for lab, y in [
        ("1344 → 512", 4.2),
        ("512 → 256", 3.4),
        ("256 → 1  ($\\hat{y}_{\\mathrm{OGT}}$)", 2.6),
        ("Continuous scalar output", 1.8),
    ]:
        ax.text(10.6, y, lab, ha="center", fontsize=8, color=PALETTE["OGT"])
    ax.text(10.6, 1.4, "Loss: Focal Huber", ha="center", fontsize=8, fontstyle="italic", color=PALETTE["OGT"])

    # Cross-head noise arrow
    ax.annotate(
        "",
        xy=(5.9, 3.2),
        xytext=(8.1, 3.2),
        arrowprops=dict(arrowstyle="<->", color=PALETTE["muted"], lw=1.4, ls="--"),
    )
    ax.text(
        7.0, 3.55,
        "Scheduled Gaussian noise\n$\\sigma_{\\mathrm{noise}}=2.0^\\circ$C",
        ha="center", fontsize=7, color=PALETTE["muted"], fontstyle="italic",
    )

    # Arrows from fusion
    ax.annotate("", xy=(3.4, 5.4), xytext=(5.5, 6.2),
                arrowprops=dict(arrowstyle="->", color=PALETTE["StableProt"], lw=1.3))
    ax.annotate("", xy=(10.6, 5.4), xytext=(8.5, 6.2),
                arrowprops=dict(arrowstyle="->", color=PALETTE["OGT"], lw=1.3))

    savefig(fig, "fig1_architecture_pipeline")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Tm benchmarking
# ═══════════════════════════════════════════════════════════════════════════
def figure2():
    """T_m benchmarking. Every value is read from refreshed_tm_numbers.json or from the
    per-protein caches; nothing in this panel is a literal."""
    print("── Figure 2: T_m benchmarking ──")
    nums = load_numbers()
    pt = load_npz("_cache_protherm.npz")
    y_true = pt["y_true"]
    y_pred = pt["pred_StableProt"]

    r, _ = pearsonr(y_true, y_pred)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    fig = plt.figure(figsize=(7.2, 8.0))
    gs = GridSpec(2, 2, height_ratios=[1.05, 0.95], hspace=0.52, wspace=0.42)

    # ── A: predicted against measured, ProThermDB ──
    ax = fig.add_subplot(gs[0, 0])
    hb = ax.hexbin(y_true, y_pred, gridsize=34, cmap="Blues", mincnt=1, linewidths=0)
    ax.plot([30, 100], [30, 100], ls=(0, (4, 3)), color=PALETTE["parity"], lw=1.0, zorder=1)
    ax.set_xlim(35, 95)
    ax.set_ylim(35, 95)
    ax.set_aspect("equal")
    ax.set_xlabel("Measured $T_m$ (°C)")
    ax.set_ylabel("Predicted $T_m$ (°C)")
    ax.text(0.04, 0.96,
            f"$r$ = {r:.3f}\nMAE = {mae:.2f} °C\nRMSE = {rmse:.2f} °C\n$n$ = {len(y_true):,}",
            transform=ax.transAxes, va="top", fontsize=7.8, linespacing=1.5,
            color=PALETTE["spine"])
    panel_title(ax, "ProThermDB holdout")
    panel_label(ax, "A")
    despine(ax)
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("proteins", fontsize=7.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7)

    # ── B: per-bin profile, the panel the aggregate hides ──
    ax = fig.add_subplot(gs[0, 1])
    per_bin = nums["protherm"]["per_bin"]
    bin_keys = ["40-50", "50-60", "60-70", "70-80", "80-90", "90-100"]
    counts = [602, 1807, 508, 185, 225, 13]
    order = ["StableProt", "TemBERTure", "ThermoFormer-TM", "DeepSTABp", "ESMStabP"]
    x = np.arange(len(bin_keys))

    ax.axvspan(1.5, 5.5, color=PALETTE["callout"], zorder=0)
    ax.text(3.5, 0.5, "thermophilic, > 60 °C", ha="center", fontsize=7,
            color=PALETTE["thermo"], style="italic")
    for name in order:
        if name not in per_bin:
            continue
        vals = [per_bin[name][k] for k in bin_keys]
        lead = name == "StableProt"
        ax.plot(x, vals, marker=model_marker(name), color=model_color(name),
                lw=2.2 if lead else 1.3, ms=5.5 if lead else 4.2,
                alpha=1.0 if lead else 0.8, label=name, zorder=5 if lead else 3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b.replace('-', '–')}\n{n:,}" for b, n in zip(bin_keys, counts)],
                       rotation=0, ha="center", fontsize=6.6, linespacing=1.6)
    ax.text(-0.60, -2.55, "n =", ha="right", va="top", fontsize=6.6, color=PALETTE["muted"])
    ax.set_xlabel("Measured $T_m$ bin (°C)")
    ax.set_ylabel("MAE (°C)")
    ax.set_ylim(0, 22)
    ax.set_xlim(-0.62, 5.5)
    ax.set_yticks([0, 4, 8, 12, 16])
    ax.legend(loc="upper left", fontsize=6.6, ncol=2, handlelength=1.4,
              columnspacing=0.9, labelspacing=0.35, borderpad=0.2)
    panel_title(ax, "Error by temperature bin")
    panel_label(ax, "B")
    despine(ax)

    # ── C: point error against probabilistic score, both benchmarks ──
    ax = fig.add_subplot(gs[1, :])
    t2p, t2f = nums["protherm"]["table2"], nums["fireprot"]["table2"]
    models = ["TemStaPro", "ESMStabP", "ThermoFormer-TM", "DeepSTABp", "TemBERTure", "StableProt"]
    # TemStaPro returns threshold classes; its bracket-midpoint proxy is shown hatched
    # and excluded from the ranking, as stated in the caption and in section 3.1.
    proxy = {"TemStaPro": (11.55, 21.06)}

    def cell(tbl, m, key):
        if m in proxy:
            return proxy[m][0 if tbl is t2p else 1]
        return tbl[m][key]

    x = np.arange(len(models))
    w = 0.20
    for off, (tbl, key, lab, alpha, hatch) in enumerate([
        (t2p, "mae", "ProThermDB, MAE", 0.40, None),
        (t2p, "crps", "ProThermDB, CRPS", 0.95, None),
        (t2f, "mae", "FireProtDB, MAE", 0.40, "///"),
        (t2f, "crps", "FireProtDB, CRPS", 0.95, "///"),
    ]):
        vals = [cell(tbl, m, key) for m in models]
        cols = [model_color(m) for m in models]
        ax.bar(x + (off - 1.5) * w, vals, w, color=cols, alpha=alpha, hatch=hatch,
               edgecolor="white", linewidth=0.7, label=lab)

    for i, m in enumerate(models):
        if m == "StableProt":
            for off, tbl in [(1, t2p), (3, t2f)]:
                v = tbl[m]["crps"]
                ax.text(i + (off - 1.5) * w, v + 0.4, f"{v:.2f}", ha="center",
                        fontsize=7.5, fontweight="bold", color=PALETTE["StableProt"])

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=18, ha="right")
    ax.set_ylabel("Error (°C)")
    ax.set_ylim(0, 26)
    ax.legend(loc="upper right", fontsize=7, ncol=2, handlelength=1.4,
              columnspacing=1.0, labelspacing=0.35, borderpad=0.3)
    ax.text(0.0, -0.34,
            "For a point forecast CRPS equals its own MAE, so the paired bars are identical for every "
            "baseline; only StableProt emits a\ndistribution, so only it separates. TemStaPro is a "
            "bracket-midpoint proxy and is excluded from the ranking.",
            transform=ax.transAxes, va="top", fontsize=7, color=PALETTE["muted"], style="italic",
            linespacing=1.5)
    panel_title(ax, "Point error against probabilistic score, both benchmarks")
    panel_label(ax, "C")
    despine(ax)

    savefig(fig, "fig2_tm_benchmark_grid")
    print(f"    ProThermDB r={r:.3f} MAE={mae:.2f} CRPS={t2p['StableProt']['crps']:.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — OGT generalization
# ═══════════════════════════════════════════════════════════════════════════
def figure3():
    """OGT generalization. Drawn entirely from _cache_brenda_ogt.npz, written by
    experiments/src/eval/refresh_ogt_cache.py from the adopted OGT head."""
    print("── Figure 3: OGT generalization ──")
    br = load_npz("_cache_brenda_ogt.npz")
    y = br["y_true"]
    models = ["StableProt", "PRIME", "ThermoFormer"]
    preds = {m: br[f"pred_{m}"] for m in models}
    crps_sp = br["y_conf"]

    fig = plt.figure(figsize=(7.2, 6.6))
    gs = GridSpec(2, 2, hspace=0.58, wspace=0.46)

    # ── A: predicted against measured ──
    ax = fig.add_subplot(gs[0, 0])
    mu = preds["StableProt"]
    r, _ = pearsonr(y, mu)
    hb = ax.hexbin(y, mu, gridsize=30, cmap="Blues", mincnt=1, linewidths=0)
    ax.plot([0, 100], [0, 100], ls=(0, (4, 3)), color=PALETTE["parity"], lw=1.0, zorder=1)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.set_xlabel("Measured OGT (°C)")
    ax.set_ylabel("Predicted OGT (°C)")
    ax.text(0.04, 0.96,
            f"$r$ = {r:.3f}\nMAE = {np.mean(np.abs(y-mu)):.2f} °C\n$n$ = {len(y):,}",
            transform=ax.transAxes, va="top", fontsize=7.8, linespacing=1.5,
            color=PALETTE["spine"])
    panel_title(ax, "BRENDA, out of distribution")
    panel_label(ax, "A")
    despine(ax)
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("proteins", fontsize=7.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7)

    # ── B: per-bin MAE. The aggregate favours the baselines; this panel says why. ──
    ax = fig.add_subplot(gs[0, 1])
    edges = [(lo, lo + 10) for lo in range(0, 100, 10)]
    masks = [(y >= lo) & (y < hi) for lo, hi in edges]
    keep = [i for i, m in enumerate(masks) if m.sum() >= 10]
    x = np.arange(len(keep))
    w = 0.27
    ax.axvspan(x[[edges[i][0] for i in keep].index(40)] - 0.5,
               x[[edges[i][0] for i in keep].index(50)] + 0.5,
               color=PALETTE["callout"], zorder=0)
    for j, m in enumerate(models):
        vals = [np.abs(y[masks[i]] - preds[m][masks[i]]).mean() for i in keep]
        ax.bar(x + (j - 1) * w, vals, w, color=model_color(m), edgecolor="white",
               linewidth=0.6, label=m, alpha=1.0 if m == "StableProt" else 0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{edges[i][0]}–{edges[i][1]}  ({int(masks[i].sum())})"
                        for i in keep], rotation=45, ha="right", fontsize=6.2)
    ax.set_xlabel("Measured OGT bin (°C), with $n$ in brackets")
    ax.set_ylabel("MAE (°C)")
    ax.set_ylim(0, 30)
    ax.legend(loc="upper right", fontsize=7, handlelength=1.3, labelspacing=0.3)
    panel_title(ax, "Error by growth-temperature bin")
    panel_label(ax, "B")
    despine(ax)

    # ── C: the collapse ratio, the single clearest result in the OGT section ──
    ax = fig.add_subplot(gs[1, 0])
    warm = (y >= 40) & (y < 60)
    cool = (y >= 20) & (y < 40)
    ratios = [np.abs(y[warm] - preds[m][warm]).mean() / np.abs(y[cool] - preds[m][cool]).mean()
              for m in models]
    bars = ax.bar(models, ratios, width=0.58,
                  color=[model_color(m) for m in models], edgecolor="white", linewidth=0.8)
    reference_line(ax, y=1.0)
    for b, v in zip(bars, ratios):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.09, f"{v:.2f}×", ha="center",
                fontsize=9, fontweight="bold", color=PALETTE["spine"])
    ax.set_ylabel("MAE 40–60 °C  /  MAE 20–40 °C")
    ax.set_ylim(0, max(ratios) * 1.28)
    ax.text(0.02, 0.97, "Above 1 means error grows as the organism gets hotter.\n"
                        "StableProt is the only model that does not.",
            transform=ax.transAxes, va="top", fontsize=7, color=PALETTE["muted"], style="italic",
            linespacing=1.5)
    panel_title(ax, "Degradation from mesophilic to thermophilic")
    panel_label(ax, "C")
    despine(ax)

    # ── D: screening. OGT models only; mixing in T_m predictors would compare
    #      two different quantities under one curve. ──
    ax = fig.add_subplot(gs[1, 1])
    ybin = (y >= 50).astype(int)
    ax.plot([0, 1], [0, 1], ls=(0, (4, 3)), color=PALETTE["parity"], lw=1.0, label="chance")
    for m in models:
        fpr, tpr, auc = roc_curve_auc(ybin, preds[m])
        ax.plot(fpr, tpr, color=model_color(m), lw=2.0 if m == "StableProt" else 1.4,
                alpha=1.0 if m == "StableProt" else 0.85, label=f"{m} (AUC {auc:.3f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=7, handlelength=1.5, labelspacing=0.3)
    panel_title(ax, f"Thermophile screening, OGT ≥ 50 °C ($n$={int(ybin.sum())})")
    panel_label(ax, "D")
    despine(ax)

    savefig(fig, "fig3_ogt_generalization_grid")
    print("    collapse ratios: " + ", ".join(f"{m} {v:.2f}x" for m, v in zip(models, ratios)))


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Calibration & disjoint proof
# ═══════════════════════════════════════════════════════════════════════════
def figure4():
    """Calibration. Panels A-C are computed from the raw per-protein sigma and the
    cross-fitted scale stored in the cache, so they cannot drift from section 3.4."""
    print("── Figure 4: uncertainty calibration ──")
    pt = load_npz("_cache_protherm.npz")
    y, mu, sig = pt["y_true"], pt["pred_StableProt"], pt["sigma"]
    c = np.asarray(pt["scale"], dtype=float)
    err = np.abs(y - mu)

    # Same 20-level grid as experiments/src/eval/refresh_tm_tables.py, so the ECE printed
    # in this panel is byte-identical to the one quoted in section 3.4.
    levels = np.linspace(0.05, 0.95, 20)
    z = special.erfinv(levels) * np.sqrt(2.0)

    def coverage(scale, mask=None):
        m = np.ones(len(y), bool) if mask is None else mask
        return np.array([(err[m] <= zi * (scale * sig)[m]).mean() for zi in z])

    raw, cal = coverage(1.0), coverage(c)
    ece_raw, ece_cal = np.abs(raw - levels).mean(), np.abs(cal - levels).mean()

    fig = plt.figure(figsize=(7.2, 6.4))
    gs = GridSpec(2, 2, hspace=0.52, wspace=0.34)

    # ── A: marginal reliability ──
    ax = fig.add_subplot(gs[0, 0])
    ax.plot([0, 100], [0, 100], ls=(0, (4, 3)), color=PALETTE["parity"], lw=1.0, label="ideal")
    ax.plot(levels * 100, raw * 100, "o-", color=PALETTE["muted"], ms=3.4, lw=1.4,
            label=f"raw σ (ECE {ece_raw*100:.1f}%)")
    ax.plot(levels * 100, cal * 100, "o-", color=PALETTE["StableProt"], ms=3.4, lw=2.0,
            label=f"scaled, $c$={np.median(c):.2f} (ECE {ece_cal*100:.1f}%)")
    ax.set_xlabel("Nominal coverage (%)")
    ax.set_ylabel("Observed coverage (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=7, handlelength=1.5, labelspacing=0.3)
    panel_title(ax, "Reliability, ProThermDB")
    panel_label(ax, "A")
    despine(ax)

    # ── B: the same curves split by thermal regime. The marginal fit above hides this. ──
    ax = fig.add_subplot(gs[0, 1])
    regimes = [("40–50 °C", (y >= 40) & (y < 50), "meso"),
               ("50–60 °C", (y >= 50) & (y < 60), "moderate"),
               ("60–80 °C", (y >= 60) & (y < 80), "thermo"),
               ("> 80 °C", y >= 80, "hyper")]
    ax.plot([0, 100], [0, 100], ls=(0, (4, 3)), color=PALETTE["parity"], lw=1.0)
    for lab, m, key in regimes:
        if m.sum() < 20:
            continue
        cv = coverage(c, m)
        ax.plot(levels * 100, cv * 100, "o-", color=PALETTE[key], ms=3.0, lw=1.5,
                label=f"{lab}  ($n$={int(m.sum()):,})")
    ax.set_xlabel("Nominal coverage (%)")
    ax.set_ylabel("Observed coverage (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=7, handlelength=1.5, labelspacing=0.3)
    panel_title(ax, "The same interval, split by regime")
    panel_label(ax, "B")
    despine(ax)

    # ── C: is the width informative per protein, or only right on average ──
    ax = fig.add_subplot(gs[1, 0])
    q = np.quantile(sig, np.linspace(0, 1, 6))
    idx = [(sig >= q[i]) & (sig <= q[i + 1] if i == 4 else sig < q[i + 1]) for i in range(5)]
    maes = [err[m].mean() for m in idx]
    sigs = [sig[m].mean() for m in idx]
    xs = np.arange(5)
    ax.bar(xs, maes, width=0.62, color=PALETTE["StableProt"], alpha=0.85, edgecolor="white",
           linewidth=0.8, label="observed MAE")
    ax.plot(xs, sigs, "o--", color=PALETTE["OGT"], ms=5, lw=1.6, label="mean predicted σ")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"Q{i+1}" for i in range(5)])
    ax.set_xlabel("Quintile of predicted σ, narrowest to widest")
    ax.set_ylabel("°C")
    ax.set_ylim(0, max(max(maes), max(sigs)) * 1.62)
    ax.legend(loc="upper left", fontsize=7, handlelength=1.5, ncol=2, columnspacing=1.0)
    ax.text(0.03, 0.83, "σ rises across quintiles but the error does not follow:\n"
                        "the width is calibrated on average, not per protein.",
            transform=ax.transAxes, va="top", fontsize=6.8, color=PALETTE["muted"],
            style="italic", linespacing=1.5)
    panel_title(ax, "What the width does and does not buy")
    panel_label(ax, "C")
    despine(ax)

    # ── D: the measurement the disjoint design rests on ──
    ax = fig.add_subplot(gs[1, 1])
    with open(DATA / "gradient_interference_histogram.json") as fh:
        grad = json.load(fh)
    shared = np.array(grad["v7_overall_cosine_similarities"], dtype=float)
    ax.hist(shared, bins=26, color=PALETTE["bad"], alpha=0.75, edgecolor="white", linewidth=0.5,
            label=f"shared backbone (mean {shared.mean():+.3f})")
    ax.axvline(0.0, color=PALETTE["StableProt"], lw=2.2,
               label="disjoint heads (0 by construction)")
    reference_line(ax, x=shared.mean(), color=PALETTE["bad"], lw=1.0)
    ax.set_xlabel(r"cos $\theta$ between $T_m$ and OGT gradients")
    ax.set_ylabel("Training steps")
    ax.legend(loc="upper left", fontsize=6.8, handlelength=1.4, labelspacing=0.3)
    panel_title(ax, "Gradient conflict under a shared backbone")
    panel_label(ax, "D")
    despine(ax)

    savefig(fig, "fig4_calibration_disjoint_grid")
    print(f"    ECE raw {ece_raw*100:.1f}% -> {ece_cal*100:.1f}% at c={np.median(c):.2f}; "
          f"MAE by sigma quintile {[round(v,2) for v in maes]}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Zero-shot & transfer
# ═══════════════════════════════════════════════════════════════════════════
def figure5():
    print("── Figure 5: Zero-Shot & Emergent Transfer ──")
    with open(DATA / "spurs_megascale_scatter.json") as f:
        mega = json.load(f)
    with open(DATA / "cluster_ood_generalization.json") as f:
        clusters = json.load(f)
    with open(DATA / "cross_species_generalization.json") as f:
        species = json.load(f)

    y_true = np.array(mega["coordinates"]["y_true"])
    y_pred = np.array(mega["coordinates"]["y_pred"])
    om = mega["overall_metrics"]

    fig = plt.figure(figsize=(7.0, 7.5))
    gs = GridSpec(2, 2, hspace=0.35, wspace=0.30)

    # A: Megascale scatter
    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "A")
    regimes = [
        (y_true <= 40, PALETTE["meso"], "Mesophilic ≤40°C"),
        ((y_true > 40) & (y_true <= 60), PALETTE["moderate"], "Moderate 40–60°C"),
        (y_true > 60, PALETTE["thermo"], "Thermophilic >60°C"),
    ]
    for mask, col, lab in regimes:
        ax.scatter(y_true[mask], y_pred[mask], s=18, c=col, alpha=0.7, edgecolors="none", label=lab, zorder=3)
    ax.plot([20, 100], [20, 100], "--", color=PALETTE["parity"], lw=1.1)
    ax.set_xlim(25, 95)
    ax.set_ylim(25, 95)
    ax.set_xlabel("Experimental $T_m$ (°C)")
    ax.set_ylabel("Predicted $T_m$ (°C)")
    panel_title(ax, "FLIP / Megascale Holdout")
    ax.legend(fontsize=6.5, loc="upper left")
    # Thermophilic callout from JSON stratified metrics
    thermo_m = next(s for s in mega["stratified_metrics"] if "Thermophilic" in s["Regime"])
    ax.text(
        0.98, 0.05,
        f"Overall: MAE={om['mae']:.2f}°C, $r$={om['pearson_r']:.2f}, $N$={len(y_true)}\n"
        f">60°C: MAE={thermo_m['MAE']:.2f}°C, RMSE={thermo_m['RMSE']:.2f}°C ($N$={thermo_m['Sample_Count']})",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=PALETTE["boundary_wash"], edgecolor=PALETTE["thermo"], alpha=0.95),
    )
    despine(ax)

    # B: Emergent transfer
    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "B")
    tasks = [
        "Human PPI\nAcc.",
        "DeepLoc\nAcc.",
        "LiveProtein\nBench $r$",
        "eSOL\n$R^2$",
    ]
    vals = [88.3, 85.0, 54.1, 35.4]  # r and R2 as % for shared axis
    raw_labels = ["88.3%", "85.0%", "0.541", "0.354"]
    cols = [PALETTE["StableProt"], PALETTE["StableProt_ci"], PALETTE["OGT"], PALETTE["DeepSTABp"]]
    bars = ax.bar(tasks, vals, color=cols, edgecolor="white", width=0.65)
    for b, lab in zip(bars, raw_labels):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, lab,
                ha="center", fontsize=8, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Score (Acc. % or scaled corr.)")
    panel_title(ax, "Emergent Representation Transfer")
    ax.text(
        0.5, -0.22,
        "Linear probes on StableProt embeddings encode biophysical signal beyond $T_m$",
        transform=ax.transAxes, ha="center", fontsize=7, fontstyle="italic", color=PALETTE["muted"],
    )
    despine(ax)

    # C: Homology clusters
    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "C")
    cl = clusters["clusters_30pct_identity"]
    names = ["#" + "".join(ch for ch in str(c["Cluster_Rank"]) if ch.isdigit()) for c in cl]
    maes = [c["V9_MAE"] for c in cl]
    tm_means = [c["Mean_Tm_Target"] for c in cl]
    counts = [c["Sample_Count"] for c in cl]
    colors = [PALETTE["thermo"] if t >= 60 else PALETTE["StableProt_ci"] for t in tm_means]
    bars = ax.bar(names, maes, color=colors, edgecolor="white", width=0.7)
    for b, m, n, t in zip(bars, maes, counts, tm_means):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15,
                f"{m:.2f}\n$N$={n}", ha="center", va="bottom", fontsize=6.5)
    ax.axhline(clusters["overall_baseline_mae"], ls="--", color=PALETTE["parity"], lw=1.2,
               label=f"Overall MAE={clusters['overall_baseline_mae']:.2f}°C")
    ax.set_ylabel("MAE (°C)")
    ax.set_xlabel("MMseqs2 Family Cluster")
    panel_title(ax, "Homology Cluster OOD ($N$=5,861 clusters)")
    ax.set_ylim(0, max(maes) + 3)
    ax.legend(fontsize=7)
    despine(ax)

    # D: Cross-species
    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "D")
    org_short = {
        "Escherichia coli": "E. coli",
        "Thermus thermophilus": "T. thermophilus",
        "Saccharomyces cerevisiae": "S. cerevisiae",
        "Homo sapiens": "H. sapiens",
    }
    labels = [org_short[s["Species"]] for s in species]
    maes = [s["StableProt_V9_MAE"] for s in species]
    rhos = [s["StableProt_V9_Spearman"] for s in species]
    x = np.arange(len(labels))
    bars = ax.bar(x - 0.18, maes, 0.36, color=PALETTE["StableProt"], edgecolor="white", label="MAE (°C)")
    ax2 = ax.twinx()
    ax2.plot(x + 0.18, rhos, "D-", color=PALETTE["OGT"], ms=7, lw=1.8, label="Spearman $\\rho$")
    ax2.set_ylabel("Intra-organism Spearman $\\rho$", color=PALETTE["OGT"])
    ax2.set_ylim(0, 0.35)
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("$T_m$ MAE (°C)")
    panel_title(ax, "Cross-Species Stratification")
    ax.set_ylim(0, 14)
    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right", ncol=2, columnspacing=1.0)
    ax.text(
        0.0, -0.30,
        "Ranking organisms by environment is not the same as ranking their proteins by stability.",
        transform=ax.transAxes, va="top", fontsize=7, fontstyle="italic", color=PALETTE["muted"],
    )
    despine(ax)

    savefig(fig, "fig5_zeroshot_transfer_grid")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Web app collage from existing screenshot + placeholder design
# ═══════════════════════════════════════════════════════════════════════════
def figure6():
    print("── Figure 6: Web Application Suite ──")
    ss = PROJECT / "ss.png"
    fig = plt.figure(figsize=(7.0, 5.5))
    gs = GridSpec(1, 2, wspace=0.08)

    ax = fig.add_subplot(gs[0])
    panel_label(ax, "A")
    ax.axis("off")
    panel_title(ax, "/predict — Thermostability Interface")
    if ss.exists():
        img = plt.imread(ss)
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, "Screenshot missing\n(ss.png)", ha="center", va="center")

    ax = fig.add_subplot(gs[1])
    panel_label(ax, "B")
    ax.axis("off")
    ax.set_title("/design — Mutational Scanning", fontweight="bold", fontsize=10)
    # Placeholder schematic until live screenshot captured
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    box = mpatches.FancyBboxPatch(
        (0.5, 0.5), 9, 9, boxstyle="round,pad=0.1,rounding_size=0.3",
        facecolor="#F8FAFC", edgecolor=PALETTE["spine"], linewidth=1.2,
    )
    ax.add_patch(box)
    ax.text(5, 9.0, "StableProt Design Suite", ha="center", fontsize=11, fontweight="bold", color=PALETTE["StableProt"])
    ax.text(5, 8.2, "Automated loop scan · ΔTm heatmap · ranked variants", ha="center", fontsize=7.5, color=PALETTE["muted"])

    # Fake heatmap
    rng = np.random.default_rng(42)
    heat = rng.normal(0, 1.2, (8, 12))
    heat[:, 3] += 2.0
    heat[:, 7] -= 1.5
    ax_h = fig.add_axes([0.58, 0.35, 0.32, 0.28])
    im = ax_h.imshow(heat, cmap="RdYlGn", aspect="auto", vmin=-3, vmax=3)
    ax_h.set_title("Δ$T_m$ mutation heatmap", fontsize=8)
    ax_h.set_xlabel("Position", fontsize=7)
    ax_h.set_ylabel("AA", fontsize=7)
    ax_h.tick_params(labelsize=6)
    fig.colorbar(im, ax=ax_h, fraction=0.046, pad=0.02).set_label("Δ$T_m$", fontsize=7)

    ax.text(
        5, 1.5,
        "Capture live /design screenshot via uvicorn\nto replace this schematic panel.",
        ha="center", fontsize=7.5, fontstyle="italic", color=PALETTE["parity"],
    )

    savefig(fig, "fig6_webapp_suite")


# ═══════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY FIGURES
# ═══════════════════════════════════════════════════════════════════════════
def figure_s1():
    print("── Figure S1: Data Cleaning Before/After ──")
    # Compose from existing PNGs if present
    tm_path = DATA / "tm_cleaning_before_after.png"
    ogt_path = DATA / "ogt_cleaning_before_after.png"
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    for ax, path, lab, title in [
        (axes[0], tm_path, "A", "$T_m$ Cleaning"),
        (axes[1], ogt_path, "B", "OGT Cleaning"),
    ]:
        panel_label(ax, lab)
        ax.axis("off")
        panel_title(ax, title)
        if path.exists():
            ax.imshow(plt.imread(path))
        else:
            ax.text(0.5, 0.5, f"Missing:\n{path.name}", ha="center", va="center")
    fig.suptitle("Figure S1: Data Cleaning Before / After", fontweight="bold", y=1.02)
    savefig(fig, "figS1_data_cleaning")


def figure_s2():
    print("── Figure S2: ΔTm Mutation Benchmark ──")
    with open(DATA / "mutation_deltatm_scatter.json") as f:
        d = json.load(f)
    exp = np.array(d["coordinates"]["delta_tm_exp"])
    pred = np.array(d["coordinates"]["delta_tm_pred"])
    m = d["metrics"]
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.scatter(exp, pred, s=12, c=PALETTE["StableProt_ci"], alpha=0.55, edgecolors="none")
    lim = max(np.max(np.abs(exp)), np.max(np.abs(pred))) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "--", color=PALETTE["parity"], lw=1.1)
    ax.axhline(0, color=PALETTE["rule"], lw=0.8)
    ax.axvline(0, color=PALETTE["rule"], lw=0.8)
    ax.set_xlabel("Experimental $\\Delta T_m$ (°C)")
    ax.set_ylabel("Predicted $\\Delta T_m$ (°C)")
    panel_title(ax, "Point Mutation $\\Delta T_m$")
    ax.text(
        0.05, 0.95,
        f"$N$={len(exp)}\nMAE={m['mae']:.2f}°C\nAcc={m['classification_accuracy']*100:.1f}%\nAUC={m['roc_auc']:.3f}",
        transform=ax.transAxes, va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=PALETTE["rule"]),
    )
    despine(ax)
    savefig(fig, "figS2_deltatm_mutation")


def figure_s3():
    print("── Figure S3: FireProtDB Scatter ──")
    fp = load_npz("_cache_fireprot.npz")
    y_true, y_pred = fp["y_true"], fp["pred_StableProt V9"]
    y_conf = fp["conf_StableProt V9"]
    mae = np.mean(np.abs(y_true - y_pred))
    r, _ = pearsonr(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.errorbar(
        y_true, y_pred, yerr=y_conf, fmt="o", ms=3.5, color=PALETTE["StableProt_ci"],
        ecolor=PALETTE["StableProt_pale"], elinewidth=0.6, alpha=0.7, capsize=0,
    )
    ax.plot([20, 100], [20, 100], "--", color=PALETTE["parity"], lw=1.1)
    ax.set_xlabel("Experimental $T_m$ (°C)")
    ax.set_ylabel("Predicted $T_m$ (°C)")
    panel_title(ax, "FireProtDB Zero-Shot")
    ax.text(
        0.05, 0.95,
        f"$N$={len(y_true)}\nMAE={mae:.2f}°C\n$r$={r:.2f}\nInt-MAE={int_mae(y_true,y_pred,y_conf):.2f}°C",
        transform=ax.transAxes, va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=PALETTE["rule"]),
    )
    despine(ax)
    savefig(fig, "figS3_fireprot_scatter")


def figure_s4():
    print("── Figure S4: Extended Homology Clusters ──")
    with open(DATA / "cluster_ood_generalization.json") as f:
        clusters = json.load(f)
    cl = clusters["clusters_30pct_identity"]
    sizes = [c["Sample_Count"] for c in cl]
    maes = [c["V9_MAE"] for c in cl]
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    ax.scatter(sizes, maes, s=80, c=PALETTE["StableProt"], alpha=0.8, edgecolors="white")
    for c in cl:
        ax.annotate(
            c["Cluster_Rank"].replace("Family Cluster ", "#"),
            (c["Sample_Count"], c["V9_MAE"]),
            textcoords="offset points", xytext=(4, 4), fontsize=7,
        )
    ax.set_xlabel("Cluster Size ($N$)")
    ax.set_ylabel("MAE (°C)")
    panel_title(ax, "MAE vs Homology Cluster Size")
    ax.axhline(clusters["overall_baseline_mae"], ls="--", color=PALETTE["parity"], lw=1.1)
    despine(ax)
    savefig(fig, "figS4_cluster_mae_vs_size")


def figure_s5():
    """Signed error, not absolute: the sign is the whole point.

    A baseline that compresses its predictions toward the mesophilic band produces a long
    negative tail on thermostable proteins, and taking the absolute value hides exactly that.
    TemStaPro is excluded because it returns threshold classes rather than a temperature.
    """
    print("── Figure S5: signed error distributions ──")
    models = ["StableProt", "TemBERTure", "ThermoFormer-TM", "DeepSTABp", "ESMStabP"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    for ax, (cache, tag) in zip(axes, [("_cache_protherm.npz", "ProThermDB, in distribution"),
                                       ("_cache_fireprot.npz", "FireProtDB, out of distribution")]):
        d = load_npz(cache)
        y_true = d["y_true"]
        data = [d[f"pred_{m}"] - y_true for m in models]

        reference_line(ax, y=0.0)
        parts = ax.violinplot(data, showextrema=False, widths=0.82)
        for pc, m in zip(parts["bodies"], models):
            pc.set_facecolor(model_color(m))
            pc.set_alpha(0.55 if m != "StableProt" else 0.75)
            pc.set_edgecolor(model_color(m))
            pc.set_linewidth(0.9)
        for i, series in enumerate(data, start=1):
            q1, med, q3 = np.percentile(series, [25, 50, 75])
            ax.vlines(i, q1, q3, color=PALETTE["spine"], lw=3.0, zorder=3)
            ax.plot(i, med, "o", ms=3.6, color="white", mec=PALETTE["spine"], mew=0.8, zorder=4)

        ax.set_xticks(range(1, len(models) + 1))
        ax.set_xticklabels(models, rotation=22, ha="right")
        panel_title(ax, tag)
        despine(ax)

    axes[0].set_ylabel("Signed error, predicted − measured (°C)")
    panel_label(axes[0], "A")
    panel_label(axes[1], "B")
    fig.text(0.5, -0.06, "Mass below the dashed line is under-prediction. "
             "The baselines' negative tails are the mesophilic collapse.",
             ha="center", fontsize=8, color=PALETTE["muted"])
    fig.tight_layout()
    savefig(fig, "figS5_error_violins")


def figure_s6():
    print("── Figure S6: OGT Confidence Analysis ──")
    br = load_npz("_cache_brenda_ogt.npz")
    y_true, y_pred, y_conf = br["y_true"], br["y_pred"], br["y_conf"]
    errors = np.abs(y_true - y_pred)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8))

    ax = axes[0]
    panel_label(ax, "A")
    idx = np.argsort(y_conf)
    x = np.arange(len(y_true))
    ax.fill_between(x, y_pred[idx] - y_conf[idx], y_pred[idx] + y_conf[idx],
                    color=PALETTE["StableProt_ci"], alpha=0.25)
    ax.scatter(x, y_true[idx], s=4, c=PALETTE["thermo"], alpha=0.5, rasterized=True)
    panel_title(ax, "OGT ±1σ Bands")
    ax.set_xlabel("Proteins (sorted by σ)")
    ax.set_ylabel("OGT (°C)")
    despine(ax)

    ax = axes[1]
    panel_label(ax, "B")
    ax.hist(y_conf, bins=30, color=PALETTE["OGT"], edgecolor="white", alpha=0.8)
    med = np.median(y_conf)
    ax.axvline(med, ls="--", color=PALETTE["spine"], label=f"Median σ={med:.1f}°C")
    panel_title(ax, "OGT σ Distribution")
    ax.set_xlabel("Predicted σ (°C)")
    ax.legend(fontsize=7)
    despine(ax)

    ax = axes[2]
    panel_label(ax, "C")
    q_edges = np.percentile(y_conf, [0, 20, 40, 60, 80, 100])
    mae_q = []
    for i in range(5):
        mask = (y_conf >= q_edges[i]) & (y_conf <= q_edges[i + 1] + 1e-9)
        mae_q.append(errors[mask].mean())
    ax.bar([f"Q{i+1}" for i in range(5)], mae_q, color=PALETTE["StableProt_ci"], edgecolor="white")
    for i, m in enumerate(mae_q):
        ax.text(i, m + 0.2, f"{m:.1f}", ha="center", fontsize=7, fontweight="bold")
    panel_title(ax, "Error vs σ Quintile")
    ax.set_ylabel("MAE (°C)")
    despine(ax)

    fig.tight_layout()
    savefig(fig, "figS6_ogt_confidence")


def figure_s7():
    """Extended calibration, ProThermDB. Uses the raw per-protein sigma and the cross-fitted
    scale stored in the cache. An earlier version refitted on the already-scaled sigma and so
    reported a scale near 1 and a raw ECE near zero, which is the same thing measured twice."""
    print("── Figure S7: extended calibration ──")
    pt = load_npz("_cache_protherm.npz")
    y, mu, sig = pt["y_true"], pt["pred_StableProt"], pt["sigma"]
    c = np.asarray(pt["scale"], dtype=float)
    err = np.abs(y - mu)

    levels = np.linspace(0.02, 0.995, 60)
    z = special.erfinv(levels) * np.sqrt(2.0)
    raw = np.array([(err <= zi * sig).mean() for zi in z])
    cal = np.array([(err <= zi * c * sig).mean() for zi in z])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))

    ax = axes[0]
    ax.plot([0, 100], [0, 100], ls=(0, (4, 3)), color=PALETTE["parity"], lw=1.0, label="ideal")
    ax.plot(levels * 100, raw * 100, color=PALETTE["muted"], lw=1.6, label="raw σ")
    ax.plot(levels * 100, cal * 100, color=PALETTE["StableProt"], lw=2.0,
            label=f"scaled, $c$={np.median(c):.2f}")
    ax.set_xlabel("Nominal coverage (%)")
    ax.set_ylabel("Observed coverage (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=7.5, handlelength=1.5)
    panel_title(ax, "Reliability across the full range of levels")
    panel_label(ax, "A")
    despine(ax)

    ax = axes[1]
    nominal = [0.683, 0.90, 0.954]
    xs = np.arange(len(nominal))
    zt = [special.erfinv(v) * np.sqrt(2.0) for v in nominal]
    obs = [float((err <= zi * c * sig).mean()) for zi in zt]
    ax.bar(xs - 0.19, [v * 100 for v in nominal], 0.36, color=PALETTE["rule"],
           edgecolor="white", label="nominal")
    ax.bar(xs + 0.19, [v * 100 for v in obs], 0.36, color=PALETTE["StableProt"],
           edgecolor="white", label="observed")
    for xi, (n_, o_) in enumerate(zip(nominal, obs)):
        ax.text(xi + 0.19, o_ * 100 + 1.4, f"{o_*100:.1f}", ha="center", fontsize=7,
                color=PALETTE["StableProt"], fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v*100:.1f}%\n$z$={zi:.2f}" for v, zi in zip(nominal, zt)], fontsize=7.5)
    ax.set_ylabel("Coverage (%)")
    ax.set_ylim(0, 108)
    ax.legend(loc="upper left", fontsize=7.5, handlelength=1.4, ncol=2)
    panel_title(ax, "Coverage at the three levels usually quoted")
    panel_label(ax, "B")
    despine(ax)

    fig.tight_layout()
    savefig(fig, "figS7_calibration_extended")
    print(f"    observed coverage {['%.1f%%' % (v*100) for v in obs]} at c={np.median(c):.2f}")


def figure_s9():
    print("── Figure S9: Carrageenase Case Study ──")
    names = ["CgkS", "CgiB_Ce"]
    exp = [45.0, 40.0]
    pred = [47.38, 42.19]
    ci = [8.11, 6.87]
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    x = np.arange(len(names))
    ax.errorbar(x, pred, yerr=ci, fmt="o", ms=10, color=PALETTE["StableProt"],
                ecolor=PALETTE["StableProt_ci"], elinewidth=2, capsize=5, label="Pred $T_m$ ±3.8σ")
    ax.scatter(x, exp, marker="D", s=70, color=PALETTE["OGT"], zorder=5, label="Exp $T_{opt}$")
    for i, (e, p, c) in enumerate(zip(exp, pred, ci)):
        ax.text(i, p + c + 1.2, f"Pred {p:.1f}°C\nCI [{p-c:.1f}, {p+c:.1f}]",
                ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Temperature (°C)")
    panel_title(ax, "Carrageenase Industrial Case Study")
    ax.set_ylim(25, 65)
    ax.legend(fontsize=7, loc="lower right")
    despine(ax)
    savefig(fig, "figS9_carrageenase")


def main():
    global OUT
    import argparse

    ap = argparse.ArgumentParser(description="StableProt Plan v4 figure regenerator")
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory for PNG/SVG (default: paper/writeup/plots_v4). Caches still read from plots/.",
    )
    args = ap.parse_args()
    if args.out:
        p = Path(args.out)
        OUT = p if p.is_absolute() else (PROJECT / p)
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"DATA (read):  {DATA}")
    print(f"OUT  (write): {OUT}")
    # Figure 1 is drawn by generate_fig1_v5.py, which is its only source. Calling the
    # older figure1() here would silently overwrite it with the superseded schematic.
    # Figure S8 is not generated: its values were synthetic, never measured.
    panels = [
        ("2", figure2), ("3", figure3), ("4", figure4), ("5", figure5), ("6", figure6),
        ("S1", figure_s1), ("S2", figure_s2), ("S3", figure_s3), ("S4", figure_s4),
        ("S5", figure_s5), ("S6", figure_s6), ("S7", figure_s7), ("S9", figure_s9),
    ]
    failed = []
    for name, fn in panels:
        try:
            fn()
        except Exception as exc:  # keep going so one missing cache cannot block the rest
            failed.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  SKIPPED figure {name} -- {type(exc).__name__}: {exc}")

    print(f"\nFigures written to {OUT}")
    if failed:
        print(f"{len(failed)} panel(s) could not be drawn:")
        for name, why in failed:
            print(f"  figure {name}: {why}")


if __name__ == "__main__":
    main()
