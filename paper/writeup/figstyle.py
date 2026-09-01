"""Single source of figure style for every panel in the manuscript and supplement.

Colour language (from images_inspiration, especially the Tm density figures):
  * cold → hot is blue → coral. That ramp is for temperature, bins, and regimes.
  * StableProt is ink navy, the one colour that means "this model".
  * TemBERTure is coral, the rival, on the warm side of the same ramp.
  * Every other model is muted so it recedes.
  * Grey is reference: identity lines, raw-σ, shared-backbone.

  from figstyle import PALETTE, MARKERS, apply, savefig, panel_label, despine, model_color
  from figstyle import thermal_cmap, ink_cmap, error_cmap, tm_hexbin
  apply()
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ── colour ────────────────────────────────────────────────────────────────────
PALETTE = {
    # models — ink navy vs coral rival, everyone else muted
    "StableProt": "#1F4E79",
    "StableProt_ci": "#5B93C5",
    "StableProt_pale": "#C5D9EC",
    "StableProt_wash": "#EEF4F9",
    "TemBERTure": "#C44E3A",
    "DeepSTABp": "#4F9A8A",
    "ESMStabP": "#D4A054",
    "ThermoFormer": "#6B5B95",
    "ThermoFormer-TM": "#6B5B95",
    "TemStaPro": "#8D949C",
    "PRIME": "#B56A5A",
    "Pro-PRIME": "#B56A5A",
    # OGT head sits on the warm side, distinct from TemBERTure
    "OGT": "#D0894B",
    "OGT_pale": "#F3D7B0",
    "OGT_wash": "#FBF4EA",
    # temperature ramp, cold to hot — bins, regimes, holdout colouring
    "psychro": "#3E7CB1",
    "meso": "#5B9A8A",
    "moderate": "#D4A054",
    "thermo": "#C44E3A",
    "hyper": "#8C2F2B",
    # structure
    "spine": "#1C1C1C",
    "muted": "#5C6570",
    "neutral": "#F3F5F7",
    "rule": "#E2E6EA",
    "parity": "#8B9199",
    "callout": "#F8E4DC",
    "boundary": "#8C2F2B",
    "boundary_wash": "#F8EDE9",
    "good": "#1F4E79",
    "bad": "#C44E3A",
}

TEMP_RAMP = [PALETTE[k] for k in ("psychro", "meso", "moderate", "thermo", "hyper")]
INK_RAMP = ["#1F4E79", "#3A6E9A", "#5B93C5", "#8FB4D0"]

MARKERS = {
    "StableProt": "o",
    "TemBERTure": "^",
    "DeepSTABp": "s",
    "ESMStabP": "D",
    "ThermoFormer": "P",
    "ThermoFormer-TM": "P",
    "TemStaPro": "v",
    "PRIME": "p",
    "Pro-PRIME": "p",
}

MODEL_ORDER = [
    "StableProt", "TemBERTure", "ThermoFormer-TM", "DeepSTABp", "ESMStabP", "TemStaPro", "PRIME",
]

OUT = Path(__file__).resolve().parent / "plots_v4"

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.titleweight": "normal",
    "axes.titlepad": 8,
    "axes.labelsize": 9.5,
    "axes.labelcolor": PALETTE["spine"],
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.color": PALETTE["spine"],
    "ytick.color": PALETTE["spine"],
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.fontsize": 8,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.85,
    "axes.edgecolor": PALETTE["spine"],
    "axes.grid": False,
    "axes.axisbelow": True,
    "lines.linewidth": 1.7,
    "lines.markersize": 5,
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "#FFFFFF",
    "svg.fonttype": "none",
}


def thermal_cmap():
    """Blue (cold) → coral (hot). For measured Tm/OGT, regime curves, bin washes."""
    return LinearSegmentedColormap.from_list("sp_thermal", [
        "#3E7CB1", "#6BA3C4", "#C5D4A3", "#E0A85E", "#C44E3A", "#8C2F2B",
    ])


def ink_cmap():
    """Pale wash → ink navy. For density hexbins when colour is count, not temperature."""
    return LinearSegmentedColormap.from_list("sp_ink", [
        "#F4F7FA", "#D4E4F0", "#8FB4D0", "#4A7FA8", "#1F4E79",
    ])


def error_cmap():
    """Navy (low error) → coral (high error). Heatmaps of MAE."""
    return LinearSegmentedColormap.from_list("sp_error", [
        "#1F4E79", "#6FA3C7", "#F4E4C1", "#E0A85E", "#C44E3A",
    ])


def tm_hexbin(ax, x, y, c=None, gridsize=34, vmin=35, vmax=95, mincnt=1):
    """Holdout hexbin. If `c` is given it is mean-reduced onto the thermal ramp."""
    kw = dict(gridsize=gridsize, mincnt=mincnt, linewidths=0.0)
    if c is None:
        return ax.hexbin(x, y, cmap=ink_cmap(), **kw)
    return ax.hexbin(
        x, y, C=c, reduce_C_function=np.nanmean,
        cmap=thermal_cmap(), vmin=vmin, vmax=vmax, **kw,
    )


def tm_colors(values, vmin=35, vmax=95):
    """Map temperatures onto the thermal ramp. Returns an (n, 4) RGBA array."""
    cmap = thermal_cmap()
    t = (np.clip(np.asarray(values, dtype=float), vmin, vmax) - vmin) / (vmax - vmin)
    return cmap(t)


def style_colorbar(cb, label, size=7.5):
    cb.set_label(label, fontsize=size)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7)
    return cb


def apply():
    matplotlib.use("Agg")
    plt.rcParams.update(RC)


def model_color(name):
    """Tolerant lookup so 'ThermoFormer (OGT ckpt, control)' resolves to its base colour."""
    if name in PALETTE:
        return PALETTE[name]
    for key in sorted(PALETTE, key=len, reverse=True):
        if name.startswith(key):
            return PALETTE[key]
    return PALETTE["muted"]


def model_marker(name):
    if name in MARKERS:
        return MARKERS[name]
    for key in sorted(MARKERS, key=len, reverse=True):
        if name.startswith(key):
            return MARKERS[key]
    return "o"


def despine(ax, left=True, bottom=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    ax.spines["left"].set_color(PALETTE["spine"])
    ax.spines["bottom"].set_color(PALETTE["spine"])
    ax.grid(False)


def panel_label(ax, label, dx=-30, dy=14):
    ax.annotate(label, xy=(0, 1), xycoords="axes fraction", xytext=(dx, dy),
                textcoords="offset points", fontsize=11.5, fontweight="bold",
                va="baseline", ha="left", color=PALETTE["spine"], annotation_clip=False)


def panel_title(ax, text):
    ax.set_title(text, loc="left", pad=10, fontsize=9.5, color=PALETTE["spine"])


def reference_line(ax, y=None, x=None, **kw):
    style = dict(color=PALETTE["parity"], lw=0.9, ls=(0, (4, 3)), zorder=0)
    style.update(kw)
    if y is not None:
        ax.axhline(y, **style)
    if x is not None:
        ax.axvline(x, **style)


def savefig(fig, stem, out=None):
    out = Path(out) if out else OUT
    out.mkdir(parents=True, exist_ok=True)
    for ax in fig.get_axes():
        ax.grid(False)
    fig.savefig(out / f"{stem}.png", dpi=300)
    fig.savefig(out / f"{stem}.svg")
    plt.close(fig)
    print(f"  saved {stem}.png / .svg")
