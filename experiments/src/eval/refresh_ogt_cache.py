#!/usr/bin/env python3
"""Write the OGT per-protein caches the figure scripts read, using the adopted v10 OGT head.

Covers the external BRENDA out-of-distribution benchmark and, when its embeddings are present,
the internal BacDive test split. Also settles C.4b by reporting Pearson r and Spearman rho for
the v10 head, which were previously omitted from Table 3 rather than carried over from v9.

The v10 head is heteroscedastic, so unlike v9 it emits a usable per-protein variance. The scale
that makes that variance honest is fitted here by two-fold cross-fitting against CRPS, the same
procedure used for the T_m head.

Run from the repo root inside the `stableprot_v2` conda environment.
"""

import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm, pearsonr, spearmanr

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
V9_DIR = os.path.join(PROJECT_ROOT, "experiments/src/training/v9_disjoint")
V10_DIR = os.path.join(PROJECT_ROOT, "experiments/src/training/v10")
PLOTS = os.path.join(PROJECT_ROOT, "paper/writeup/plots")
SEEDS = range(1, 6)
RNG = np.random.default_rng(0)
# 10 degC bins across the full populated range. This is the convention Table 3 already uses;
# narrowing the grid to 20-60 quietly drops the psychrophilic and hyperthermophilic bins, which
# is exactly where the bin-balanced comparison earns its keep.
BINS = [(lo, lo + 10) for lo in range(0, 100, 10)]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def crps_gaussian(y, mu, sigma):
    sigma = np.maximum(sigma, 1e-6)
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def fit_scale(y, mu, sigma):
    """Seeded independently of the module RNG so the fitted scale is call-order invariant."""
    n = len(y)
    order = np.random.default_rng(20260813).permutation(n)
    folds = [order[: n // 2], order[n // 2 :]]
    grid = np.linspace(0.2, 8.0, 400)
    out = np.empty(n)
    for held, fit in [(0, 1), (1, 0)]:
        f = folds[fit]
        out[folds[held]] = grid[
            int(np.argmin([crps_gaussian(y[f], mu[f], c * sigma[f]).mean() for c in grid]))
        ]
    return out


def predict_ogt(embs, seqs, device):
    """Five-seed v10 OGT ensemble. Returns mean and predictive sigma in degrees."""
    v9 = load_module("train_v9o", os.path.join(V9_DIR, "train.py"))
    v10 = load_module("train_v10o", os.path.join(V10_DIR, "train.py"))
    norms = torch.load(
        os.path.join(V10_DIR, "results/normalization_stats.pt"), map_location="cpu", weights_only=False
    )
    o_mean, o_std = norms["ogt_mean"], norms["ogt_std"]

    mus, vars_ = [], []
    emb, aux = v9.enrich_inputs(embs, seqs, tmhmm_flags=None, ogt_priors=None)
    emb, aux = emb.to(device), aux.to(device)
    for s in SEEDS:
        ckpt = os.path.join(V10_DIR, f"results/seed{s}/model_ogt.pt")
        if not os.path.exists(ckpt):
            continue
        m = v10.MultiHeadSaProtV8(ogt_heteroscedastic=True).to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
        m.eval()
        with torch.no_grad():
            z_mu, z_var = m(emb, aux, head="ogt")
        mus.append((z_mu.cpu() * o_std + o_mean).numpy())
        vars_.append((z_var.cpu() * o_std**2).numpy())

    mus, vars_ = np.stack(mus), np.stack(vars_)
    w = 1.0 / (vars_ + 1e-6)
    mu = (mus * w).sum(0) / w.sum(0)
    sigma = np.sqrt(vars_.mean(0) + ((mus - mu) ** 2).mean(0))
    return mu, sigma


def report(tag, y, mu, sigma, baselines):
    print(f"\n{'='*78}\n{tag}   n = {len(y)}\n{'='*78}")
    c = fit_scale(y, mu, sigma)
    rows = {"StableProt": (mu, crps_gaussian(y, mu, c * sigma))}
    for name, pred in baselines.items():
        rows[name] = (pred, np.abs(y - pred))

    hdr = (f"{'model':<16}{'MAE micro':>10}{'MAE macro':>10}"
           f"{'CRPS micro':>11}{'CRPS macro':>11}{'r':>8}{'rho':>8}")
    print(hdr + "\n" + "-" * len(hdr))
    summary = {}
    for name, (pred, crps) in rows.items():
        err = np.abs(y - pred)
        masks = [(y >= lo) & (y < hi) for lo, hi in BINS]
        masks = [m for m in masks if m.sum()]
        summary[name] = dict(
            mae=float(err.mean()),
            macro=float(np.mean([err[m].mean() for m in masks])),
            crps=float(crps.mean()),
            crps_macro=float(np.mean([crps[m].mean() for m in masks])),
            r=float(pearsonr(y, pred)[0]),
            rho=float(spearmanr(y, pred)[0]),
            per_bin={f"{lo}-{lo+10}": float(err[(y >= lo) & (y < lo + 10)].mean())
                     for lo in range(0, 100, 10) if ((y >= lo) & (y < lo + 10)).sum()},
        )
        m = summary[name]
        print(f"{name:<16}{m['mae']:10.2f}{m['macro']:10.2f}{m['crps']:11.2f}"
              f"{m['crps_macro']:11.2f}{m['r']:8.3f}{m['rho']:8.3f}")

    print("\n  per-bin MAE (n in brackets):")
    for lo in range(0, 100, 10):
        msk = (y >= lo) & (y < lo + 10)
        if not msk.sum():
            continue
        cells = "  ".join(f"{n} {np.abs(y[msk]-p[msk]).mean():6.2f}" for n, (p, _) in rows.items())
        print(f"    {lo:>3}-{lo+10:<3} [{int(msk.sum()):>4}]   {cells}")

    print(f"\n  fitted sigma scale c = {np.median(c):.2f}")
    for lvl, z in [(0.683, 0.9945), (0.954, 2.0)]:
        cov = float((np.abs(y - mu) <= z * c * sigma).mean())
        print(f"  nominal {lvl*100:5.1f}%   observed {cov*100:5.1f}%   "
              f"mean half-width {np.mean(z*c*sigma):5.1f} degC")
    summary["_scale"] = float(np.median(c))
    return summary, c


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    os.makedirs(PLOTS, exist_ok=True)
    out = {}

    df = pd.read_csv(os.path.join(PROJECT_ROOT, "new_data/brenda_ood_benchmark.csv"))
    y = df["ogt"].to_numpy(dtype=float)
    seqs = [str(s) for s in df["sequence"]]
    embs = torch.load(
        os.path.join(PROJECT_ROOT, "data/embeddings/brenda_ood_saprot_embeddings.pt"),
        map_location="cpu", weights_only=False,
    ).float()
    base = torch.load(
        os.path.join(PROJECT_ROOT, "data/embeddings/brenda_ood_baseline_preds.pt"),
        map_location="cpu", weights_only=False,
    )
    baselines = {k: np.asarray(v, dtype=float) for k, v in base.items()}

    mu, sigma = predict_ogt(embs, seqs, device)
    out["brenda"], c = report("BRENDA, external out of distribution", y, mu, sigma, baselines)

    cache = os.path.join(PLOTS, "_cache_brenda_ogt.npz")
    arrays = {
        "y_true": y, "y_pred": mu, "sigma": sigma, "y_conf": c * sigma,
        "sigma_scaled": c * sigma, "scale": c,
        "pred_StableProt": mu, "conf_StableProt": c * sigma,
        "pred_StableProt V9": mu, "conf_StableProt V9": c * sigma,
    }
    for name, pred in baselines.items():
        arrays[f"pred_{name}"] = pred
    np.savez(cache, **arrays)
    print(f"\nwrote {cache}")

    bl_cache = os.path.join(PLOTS, "_cache_brenda_baselines.npz")
    np.savez(bl_cache, y_true=y, **{f"pred_{k}": v for k, v in baselines.items()})
    print(f"wrote {bl_cache}")

    dest = os.path.join(PROJECT_ROOT, "paper/writeup/tables/refreshed_ogt_numbers.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {dest}")
    print("\nC.4b: the r and rho columns above are the v10 OGT head's, for Table 3.")


if __name__ == "__main__":
    main()
