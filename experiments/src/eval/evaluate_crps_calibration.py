#!/usr/bin/env python3
"""
CRPS and calibration diagnostic (REVISION_PLAN section 3.14).

Re-scores the saved prediction files. No retraining, no model inference.

Answers four questions:
  1. What global scale `c` on sigma is correct, and by which criterion (CRPS vs ECE)?
     This resolves the 2.8-vs-3.8 inconsistency in the manuscript.
  2. What is the empirical coverage of mu +- k*sigma, and which k gives 68/90/95%?
  3. On CRPS, where does StableProt rank against baselines scored as point forecasts?
     A point forecast scored by CRPS gets exactly its MAE, so this is a fair single table.
  4. Does per-protein heteroscedastic sigma beat a single global sigma (internal ablation),
     and how would baselines fare if granted their best constant sigma (private control)?

`c` is fitted by 2-fold cross-fitting so no reported number is fitted on the data it scores.
"""

import os
import numpy as np
import torch
import scipy.special
from scipy.stats import norm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

BENCHMARKS = [
    ("ProThermDB", os.path.join(PROJECT_ROOT, "new_data/protherm_evaluation_results.pt")),
    ("FireProtDB", os.path.join(PROJECT_ROOT, "new_data/fireprot_evaluation_results.pt")),
]

# Development iterations that are not reported in the paper.
SKIP = {"V1 Baseline", "V2 Improved", "V3 Regression", "V4 Improved Regr.",
        "V5 Multi-Head (ProtT5)", "StableProt", "StableProt V7"}
OURS = "StableProt V9"

# TemStaPro emits bracket classes, not a temperature; excluded from error columns.
NOT_A_REGRESSOR = {"TemStaPro"}

RNG = np.random.default_rng(0)


def crps_gaussian(y, mu, sigma):
    """Mean CRPS of a Gaussian forecast. Tends to mean|y-mu| as sigma -> 0."""
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    z = (y - mu) / sigma
    return float(np.mean(sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))))


def crps_point(y, mu):
    return float(np.mean(np.abs(y - mu)))


def ece(y, mu, sigma, z_vals=None):
    """Matches evaluate_calibration_reliability.py: mean |empirical - expected| coverage."""
    if z_vals is None:
        z_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                           1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0])
    expected = scipy.special.erf(z_vals / np.sqrt(2.0))
    err = np.abs(y - mu)
    empirical = np.array([np.mean(err <= z * sigma) for z in z_vals])
    return float(np.mean(np.abs(empirical - expected)))


def fit_scale(y, mu, sigma, objective="crps"):
    """Golden-section-free coarse-to-fine search for the scale minimising the objective."""
    lo, hi = 0.05, 12.0
    for _ in range(4):
        grid = np.linspace(lo, hi, 240)
        if objective == "crps":
            vals = [crps_gaussian(y, mu, c * sigma) for c in grid]
        else:
            vals = [ece(y, mu, c * sigma) for c in grid]
        i = int(np.argmin(vals))
        step = grid[1] - grid[0]
        lo, hi = max(0.01, grid[i] - step), grid[i] + step
    return float(grid[i])


def cross_fitted_scale(y, mu, sigma, objective="crps", n_rep=20):
    """Fit c on one half, apply to the other, average both directions. Avoids fitting on test."""
    n = len(y)
    scales, scores = [], []
    for _ in range(n_rep):
        perm = RNG.permutation(n)
        a, b = perm[: n // 2], perm[n // 2:]
        for fit_idx, ev_idx in ((a, b), (b, a)):
            c = fit_scale(y[fit_idx], mu[fit_idx], sigma[fit_idx], objective)
            scales.append(c)
            scores.append(crps_gaussian(y[ev_idx], mu[ev_idx], c * sigma[ev_idx]))
    return float(np.mean(scales)), float(np.std(scales)), float(np.mean(scores))


def bootstrap_diff(y, mu_a, sig_a, mu_b, n_boot=2000):
    """Paired bootstrap of CRPS(ours, Gaussian) - CRPS(other, point). Negative favours ours."""
    n = len(y)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, n, n)
        diffs[i] = crps_gaussian(y[idx], mu_a[idx], sig_a[idx]) - crps_point(y[idx], mu_b[idx])
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def analyse(tag, path):
    if not os.path.exists(path):
        print(f"  [skip] missing {path}")
        return
    d = torch.load(path, map_location="cpu", weights_only=False)
    y_all = np.asarray(d["y_true"], dtype=float)
    mu_all = np.asarray(d["predictions"][OURS], dtype=float)
    sig_all = np.asarray(d["confidences"][OURS], dtype=float)

    keep = np.isfinite(y_all) & np.isfinite(mu_all) & np.isfinite(sig_all) & (sig_all > 0)
    y, mu, sig = y_all[keep], mu_all[keep], sig_all[keep]

    print("\n" + "=" * 84)
    print(f"  {tag}   n = {len(y)}")
    print("=" * 84)
    print(f"  raw sigma: mean {sig.mean():.3f}  median {np.median(sig):.3f}  "
          f"min {sig.min():.3f}  max {sig.max():.3f}")
    print(f"  point MAE {np.mean(np.abs(y - mu)):.3f}   RMSE {np.sqrt(np.mean((y - mu) ** 2)):.3f}")

    # ---- Q1: the scale, by each criterion -------------------------------------------------
    c_crps_in = fit_scale(y, mu, sig, "crps")
    c_ece_in = fit_scale(y, mu, sig, "ece")
    c_cf, c_cf_sd, crps_cf = cross_fitted_scale(y, mu, sig, "crps")
    print("\n  -- global sigma scale c --")
    print(f"     minimising CRPS (in-sample) : c = {c_crps_in:.3f}")
    print(f"     minimising ECE  (in-sample) : c = {c_ece_in:.3f}")
    print(f"     minimising CRPS (cross-fit) : c = {c_cf:.3f} +- {c_cf_sd:.3f}"
          f"   -> held-out CRPS {crps_cf:.3f}")
    print(f"     implied 95% interval multiplier 1.96*c = {1.96 * c_cf:.2f}")
    print(f"     ECE at c=1.0 {100*ece(y,mu,sig):.2f}%   c=2.8 {100*ece(y,mu,2.8*sig):.2f}%   "
          f"c=3.8 {100*ece(y,mu,3.8*sig):.2f}%   c={c_cf:.2f} {100*ece(y,mu,c_cf*sig):.2f}%")

    # ---- Q2: coverage ---------------------------------------------------------------------
    ratio = np.abs(y - mu) / sig
    print("\n  -- empirical coverage of mu +- k*sigma_raw --")
    print(f"     {'k':>6} {'coverage':>10} {'mean half-width':>17}")
    for k in (1.0, 1.96, 2.8, 3.8):
        print(f"     {k:>6.2f} {100*np.mean(ratio <= k):>9.1f}% {np.mean(k*sig):>16.2f}")
    print("     k required for target coverage:")
    for t in (0.683, 0.90, 0.95):
        print(f"       {100*t:>5.1f}% -> k = {np.quantile(ratio, t):.2f}")
    print("     with sigma_cal = c*sigma (c cross-fitted):")
    for lvl, z in (("68%", 1.0), ("90%", 1.645), ("95%", 1.96)):
        cov = np.mean(ratio <= z * c_cf)
        print(f"       nominal {lvl} (z={z}) -> empirical {100*cov:.1f}%, "
              f"mean half-width {np.mean(z*c_cf*sig):.2f} C")

    # ---- Q3: CRPS leaderboard -------------------------------------------------------------
    ours_crps = crps_gaussian(y, mu, c_cf * sig)
    rows = [(f"{OURS}  [Gaussian, c={c_cf:.2f}]", ours_crps, np.mean(np.abs(y - mu))),
            (f"{OURS}  [point only]", crps_point(y, mu), np.mean(np.abs(y - mu)))]
    for name, p in d["predictions"].items():
        if name in SKIP or name == OURS or name in NOT_A_REGRESSOR:
            continue
        p = np.asarray(p, dtype=float)
        if p.shape != y_all.shape:
            continue
        p = p[keep]
        if not np.isfinite(p).all():
            continue
        rows.append((name, crps_point(y, p), np.mean(np.abs(y - p))))

    print("\n  -- CRPS leaderboard (point forecasts: CRPS == MAE exactly) --")
    print(f"     {'model':<38} {'CRPS':>8} {'MAE':>8}")
    for n_, c_, m_ in sorted(rows, key=lambda r: r[1]):
        mark = " <--" if n_.startswith(OURS) else ""
        print(f"     {n_:<38} {c_:>8.3f} {m_:>8.3f}{mark}")

    # significance vs the strongest point baseline
    pts = [(n_, c_) for n_, c_, _ in rows if not n_.startswith(OURS)]
    if pts:
        best_name, _ = min(pts, key=lambda r: r[1])
        best_pred = np.asarray(d["predictions"][best_name], dtype=float)[keep]
        lo, hi = bootstrap_diff(y, mu, c_cf * sig, best_pred)
        verdict = "ours better" if hi < 0 else ("ours worse" if lo > 0 else "tie")
        print(f"\n     paired bootstrap, ours(CRPS) - {best_name}(MAE): "
              f"95% CI [{lo:+.3f}, {hi:+.3f}]  -> {verdict}")

    # ---- Q4a: internal ablation, per-protein sigma vs one global sigma --------------------
    const = np.full_like(sig, 1.0)
    c_const, _, crps_const = cross_fitted_scale(y, mu, const, "crps")
    print("\n  -- internal ablation: is heteroscedastic sigma earning its keep? --")
    print(f"     per-protein sigma (c={c_cf:.2f})   CRPS {crps_cf:.3f}")
    print(f"     single global sigma ({c_const:.2f} C)  CRPS {crps_const:.3f}")
    delta = crps_const - crps_cf
    print(f"     gain from heteroscedasticity: {delta:+.3f} C "
          f"({'supports' if delta > 0 else 'does NOT support'} the architecture claim)")

    # ---- Q4b: private control, baselines granted their best constant sigma ---------------
    print("\n  -- PRIVATE control (do not publish without deciding): "
          "baselines given best constant sigma --")
    for n_, _, _ in sorted(rows, key=lambda r: r[1]):
        if n_.startswith(OURS):
            continue
        p = np.asarray(d["predictions"][n_], dtype=float)[keep]
        cb, _, crps_b = cross_fitted_scale(y, p, const, "crps", n_rep=6)
        flag = "  ** beats ours **" if crps_b < ours_crps else ""
        print(f"     {n_:<30} best const sigma {cb:>6.2f} C  ->  CRPS {crps_b:.3f}{flag}")

    # ---- Int-MAE, for comparison only -----------------------------------------------------
    print("\n  -- Int-MAE (improper; reported for comparison only) --")
    for k in (1.0, 1.96, 2.8, 3.8):
        print(f"     k={k:<4} Int-MAE {np.mean(np.maximum(0.0, np.abs(y - mu) - k * sig)):.3f}")


if __name__ == "__main__":
    for tag, path in BENCHMARKS:
        analyse(tag, path)
    print()
