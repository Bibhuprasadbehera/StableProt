"""Single source of figure style for every panel in the manuscript and supplement.

Figures were previously drawn by three scripts that each defined their own palette and rcParams,
which is why the assembled set looked like it came from different papers: the OGT accent differed
between Figure 1 and Figure 3, panel labels sat at different offsets, and grids were dotted in one
script and solid in another.

Every generation script imports from here and defines nothing of its own.

  from figstyle import PALETTE, MARKERS, apply, savefig, panel_label, despine, model_color
  apply()

House rules, from the journal proofs and from what actually reads well in print:
  * no grid lines anywhere; where a reference level is genuinely needed, draw one axhline
  * top and right spines off
  * one colour per model, used consistently in every figure it appears in
  * one temperature ramp, cold to hot, used for every binned-by-temperature panel
  * panel labels bold, upper left, outside the axes, at a fixed offset
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# ── colour ────────────────────────────────────────────────────────────────────
# Model colours are chosen to stay distinguishable in greyscale and under the
# common forms of colour-vision deficiency: the two that carry the argument,
# StableProt and TemBERTure, are separated on both hue and lightness.
PALETTE = {
    # models
    "StableProt": "#1E3A5F",       # deep navy, the house colour
    "StableProt_ci": "#4A90D9",    # interval fill and calibrated curves
    "StableProt_pale": "#B8D4F0",
    "StableProt_wash": "#F0F5FA",
    "TemBERTure": "#E07B54",       # terracotta, the strongest baseline
    "DeepSTABp": "#7BAE7F",
    "ESMStabP": "#C4A35A",
    "ThermoFormer": "#9B6B9E",
    "ThermoFormer-TM": "#9B6B9E",
    "TemStaPro": "#A0A0A0",
    "PRIME": "#D4807B",
    "Pro-PRIME": "#D4807B",
    # the OGT head, one value everywhere
    "OGT": "#C2610C",
    "OGT_pale": "#FBD9A5",
    "OGT_wash": "#FFF7ED",
    # temperature ramp, cold to hot, for any panel binned by temperature
    "psychro": "#4A90D9",
    "meso": "#7BAE7F",
    "moderate": "#E4B363",
    "thermo": "#E07B54",
    "hyper": "#A63A2B",
    # structure
    "spine": "#2D2D2D",
    "muted": "#6B7280",
    "neutral": "#F3F4F6",
    "rule": "#D8DCE2",
    "parity": "#9CA3AF",       # identity lines, reference levels
    "callout": "#FDEBE4",      # shaded regions of interest, must stay pale
    "boundary": "#B91C1C",     # train/test separation marks
    "boundary_wash": "#FEF2F2",
    "good": "#2E7D5B",
    "bad": "#A63A2B",
}

TEMP_RAMP = [PALETTE[k] for k in ("psychro", "meso", "moderate", "thermo", "hyper")]

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

# Fixed drawing order so the legend reads the same in every figure.
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
    "axes.linewidth": 0.9,
    "axes.edgecolor": PALETTE["spine"],
    "axes.grid": False,          # house rule: no grids
    "axes.axisbelow": True,
    "lines.linewidth": 1.6,
    "lines.markersize": 5,
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "#FFFFFF",
    "svg.fonttype": "none",      # keep text editable in the vector output
}


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
    """Bold panel letter outside the axes, upper left.

    Offset in typographic points rather than axes fractions: a fraction offset scales with the
    width of the axes, so the letter drifted into the title on wide panels and off the canvas on
    narrow ones. Points keep it in the same place in every figure.
    """
    ax.annotate(label, xy=(0, 1), xycoords="axes fraction", xytext=(dx, dy),
                textcoords="offset points", fontsize=11.5, fontweight="bold",
                va="baseline", ha="left", color=PALETTE["spine"], annotation_clip=False)


def panel_title(ax, text):
    """Titles sit left-aligned above the axes, clear of the panel label's fixed point offset."""
    ax.set_title(text, loc="left", pad=10, fontsize=9.5, color=PALETTE["spine"])


def reference_line(ax, y=None, x=None, **kw):
    """The only permitted substitute for a grid."""
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
