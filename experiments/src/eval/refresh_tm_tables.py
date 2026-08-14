#!/usr/bin/env python3
"""C.4c: regenerate every T_m number in the manuscript under the shipped configuration.

C.4a settled that the v10 OGT head can serve both the reported OGT task and the T_m prior, so the
numbers the paper prints have to be the ones that configuration emits. Rather than re-run five
evaluation scripts, this emits everything Tables 2 and 4, the Figure 2B caption and the §3.4 prose
need, in one pass:

  * Table 2      MAE, CRPS, r, rho for every model on both benchmarks, with paired bootstrap CIs
                 on each baseline's difference from StableProt
  * Figure 2B    per-10 degC-bin MAE for every model, plus micro and macro averages
  * Table 4      raw and scaled ECE, coverage and mean half-width at 68.3% and 95.4%, interval MAE
  * section 3.4  conditional coverage by temperature regime, sigma-vs-error rank correlation,
                 MAE by sigma quintile, interval-width coefficient of variation

Baseline point forecasts have CRPS equal to their absolute error, which is how a deterministic
prediction scores under a proper probabilistic rule.

Run from the repo root inside the `stableprot_v2` conda environment.
Writes JSON to paper/writeup/tables/refreshed_tm_numbers.json.
"""

import json
import os
import sys

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr, norm

sys.path.insert(0, os.path.dirname(__file__))
from evaluate_ogt_prior_swap import (  # noqa: E402
    PROJECT_ROOT,
    contaminated_train_seqs,
    crps_gaussian,
    load_fireprot,
    load_protherm,
    predict,
    train_stats,
)

N_BOOT = 4000
RNG = np.random.default_rng(0)
BINS = [(40, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]
BASELINES = {
    "TemBERTure": "temberture",
    "DeepSTABp": "deepstabp",
    "ESMStabP": "esmstabp",
    "ThermoFormer-TM": "thermoformer_tm",
}


def kept_index_maps(train_seqs):
    """Indices into new_data/baseline_predictions.pt matching each decontaminated benchmark."""
    from Bio import SeqIO
    import pandas as pd

    df = pd.read_csv(os.path.join(PROJECT_ROOT, "new_data/prothermdb_validation.csv"))
    uids = {str(r["UniProt_ID"]) for _, r in df.iterrows() if not np.isnan(r["Tm"])}
    kept_p, idx = [], 0
    for rec in SeqIO.parse(os.path.join(PROJECT_ROOT, "new_data/prothermdb_validation.fasta"), "fasta"):
        if rec.id.split("|")[0] in uids:
            if str(rec.seq).upper() not in train_seqs:
                kept_p.append(idx)
            idx += 1

    d = torch.load(
        os.path.join(PROJECT_ROOT, "data/test_data/fireprot_holdout_saprot.pt"),
        map_location="cpu",
        weights_only=False,
    )
    kept_f = [i for i, s in enumerate(str(x) for x in d["sequences"]) if s.upper() not in train_seqs]
    return kept_p, kept_f


def fit_scale(y, mu, sigma, objective="crps"):
    """Two-fold cross-fitted scale. Returns the per-point scale and the two fold values.

    Seeded independently of the module RNG so the fold split, and therefore the fitted scale
    and every CRPS derived from it, does not depend on how many bootstrap draws happened to be
    taken before this call.
    """
    n = len(y)
    order = np.random.default_rng(20260813).permutation(n)
    folds = [order[: n // 2], order[n // 2 :]]
    grid = np.linspace(0.2, 8.0, 400)
    out, chosen = np.empty(n), []
    for held, fit in [(0, 1), (1, 0)]:
        f = folds[fit]
        if objective == "crps":
            scores = [crps_gaussian(y[f], mu[f], c * sigma[f]).mean() for c in grid]
        else:
            scores = [ece(y[f], mu[f], c * sigma[f]) for c in grid]
        c = float(grid[int(np.argmin(scores))])
        out[folds[held]] = c
        chosen.append(c)
    return out, chosen


def ece(y, mu, sigma, n_levels=20):
    """Mean absolute gap between nominal and observed central-interval coverage."""
    levels = np.linspace(0.05, 0.95, n_levels)
    z = norm.ppf(0.5 + levels / 2)
    inside = np.abs(y - mu)[None, :] <= (z[:, None] * sigma[None, :])
    return float(np.mean(np.abs(inside.mean(axis=1) - levels)))


def boot_ci(diff):
    n = len(diff)
    b = np.array([diff[RNG.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def boot_ci_corr(y, a, b, kind="pearson"):
    """Paired bootstrap on the correlation difference, baseline minus StableProt."""
    f = pearsonr if kind == "pearson" else spearmanr
    n = len(y)
    d = []
    for _ in range(N_BOOT):
        i = RNG.integers(0, n, n)
        if len(np.unique(y[i])) < 3:
            continue
        d.append(f(y[i], b[i])[0] - f(y[i], a[i])[0])
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def constant_width_crps(y, mu, sigma):
    """Best single global sigma, to test whether the per-protein sigma earns its place."""
    grid = np.linspace(0.5, 40.0, 800)
    const = min(grid, key=lambda s: crps_gaussian(y, mu, np.full_like(y, s)).mean())
    return float(const), float(crps_gaussian(y, mu, np.full_like(y, const)).mean())


def table2(tag, y, sp_mu, sp_sigma, c, baseline_preds):
    print(f"\n{'='*94}\nTABLE 2 — {tag}   n = {len(y)}\n{'='*94}")
    rows, sp_err = {}, np.abs(y - sp_mu)
    sp_crps = crps_gaussian(y, sp_mu, c * sp_sigma)
    rows["StableProt"] = dict(
        mae=float(sp_err.mean()), crps=float(sp_crps.mean()),
        r=float(pearsonr(y, sp_mu)[0]), rho=float(spearmanr(y, sp_mu)[0]),
    )
    for name, pred in baseline_preds.items():
        err = np.abs(y - pred)
        d_mae, d_crps = err - sp_err, err - sp_crps
        lo_m, hi_m = boot_ci(d_mae)
        lo_c, hi_c = boot_ci(d_crps)
        rows[name] = dict(
            mae=float(err.mean()), crps=float(err.mean()),
            r=float(pearsonr(y, pred)[0]), rho=float(spearmanr(y, pred)[0]),
            d_mae=float(d_mae.mean()), d_mae_ci=[lo_m, hi_m],
            d_crps=float(d_crps.mean()), d_crps_ci=[lo_c, hi_c],
        )

    hdr = f"{'model':<18}{'MAE':>8}{'CRPS':>8}{'r':>8}{'rho':>8}   {'dMAE vs SP [95% CI]':<28}{'verdict':<12}"
    print(hdr + "\n" + "-" * len(hdr))
    for name, m in rows.items():
        if "d_mae" not in m:
            print(f"{name:<18}{m['mae']:8.2f}{m['crps']:8.2f}{m['r']:8.3f}{m['rho']:8.3f}")
            continue
        lo, hi = m["d_mae_ci"]
        v = "tie" if lo <= 0 <= hi else ("StableProt" if lo > 0 else name)
        print(
            f"{name:<18}{m['mae']:8.2f}{m['crps']:8.2f}{m['r']:8.3f}{m['rho']:8.3f}   "
            f"{m['d_mae']:+6.2f} [{lo:+.2f}, {hi:+.2f}]{'':<7}{v:<12}"
        )
    print("\n  CRPS differences (baseline minus StableProt, positive favours StableProt):")
    for name, m in rows.items():
        if "d_crps" in m:
            lo, hi = m["d_crps_ci"]
            v = "tie" if lo <= 0 <= hi else ("StableProt" if lo > 0 else name)
            print(f"    {name:<18}{m['d_crps']:+6.2f} [{lo:+.2f}, {hi:+.2f}]   {v}")

    print("\n  correlation differences (baseline minus StableProt, positive favours the baseline):")
    for name, pred in baseline_preds.items():
        for kind, key in [("pearson", "r"), ("spearman", "rho")]:
            m, lo, hi = boot_ci_corr(y, sp_mu, pred, kind)
            rows[name][f"d_{key}"] = m
            rows[name][f"d_{key}_ci"] = [lo, hi]
            v = "tie" if lo <= 0 <= hi else ("baseline" if lo > 0 else "StableProt")
            print(f"    {name:<18}d{key:<4}{m:+7.3f} [{lo:+.3f}, {hi:+.3f}]   {v}")
    return rows


def per_bin(tag, y, sp_mu, baseline_preds):
    print(f"\n{'='*94}\nFIGURE 2B — per-bin MAE, {tag}\n{'='*94}")
    models = {"StableProt": sp_mu, **baseline_preds}
    names = list(models)
    print(f"{'bin':<12}{'n':>6}" + "".join(f"{n:>18}" for n in names))
    out = {n: {} for n in names}
    for lo, hi in BINS:
        m = (y >= lo) & (y < hi)
        if m.sum() == 0:
            continue
        cells = []
        for n in names:
            v = float(np.abs(y[m] - models[n][m]).mean())
            out[n][f"{lo}-{hi}"] = v
            cells.append(v)
        best = min(cells)
        print(
            f"{f'{lo}-{hi}':<12}{int(m.sum()):>6}"
            + "".join(f"{v:>17.2f}{'*' if v == best else ' '}" for v in cells)
        )
    print(f"{'micro':<12}{len(y):>6}" + "".join(f"{np.abs(y-models[n]).mean():>18.2f}" for n in names))
    print(f"{'macro':<12}{'':>6}" + "".join(f"{np.mean(list(out[n].values())):>18.2f}" for n in names))
    for n in names:
        out[n]["micro"] = float(np.abs(y - models[n]).mean())
        out[n]["macro"] = float(np.mean([v for k, v in out[n].items() if k != "micro"]))
    return out


def calibration(tag, y, mu, sigma, c):
    print(f"\n{'='*94}\nTABLE 4 and section 3.4 — calibration, {tag}\n{'='*94}")
    err = np.abs(y - mu)
    scaled = c * sigma
    res = {
        "ece_raw": ece(y, mu, sigma),
        "ece_scaled": ece(y, mu, scaled),
        "c_median": float(np.median(c)),
    }
    print(f"  fitted scale c        {res['c_median']:.2f}")
    print(f"  ECE raw               {res['ece_raw']*100:.1f}%")
    print(f"  ECE after scaling     {res['ece_scaled']*100:.1f}%")
    for lvl, z in [(0.683, 0.9945), (0.954, 2.0)]:
        cov = float((err <= z * scaled).mean())
        half = float(np.mean(z * scaled))
        res[f"cov_{int(lvl*1000)}"] = cov
        res[f"halfwidth_{int(lvl*1000)}"] = half
        print(f"  nominal {lvl*100:5.1f}%        observed {cov*100:5.1f}%   mean half-width {half:5.1f} degC")
    int_mae = float(np.maximum(0.0, err - 2 * scaled).mean())
    res["interval_mae"] = int_mae
    print(f"  interval MAE (95%)    {int_mae:.2f} degC")

    print("\n  conditional coverage of the nominal 68.3% interval, by measured temperature:")
    res["conditional"] = {}
    for lo, hi in [(0, 40), (40, 60), (60, 80), (80, 200)]:
        m = (y >= lo) & (y < hi)
        if m.sum() < 5:
            continue
        cov = float((err[m] <= 0.9945 * scaled[m]).mean())
        res["conditional"][f"{lo}-{hi}"] = {"n": int(m.sum()), "coverage": cov}
        print(f"    {lo:>3}-{hi:<3} degC   n = {int(m.sum()):>5}   coverage {cov*100:5.1f}%")

    const_s, const_crps = constant_width_crps(y, mu, sigma)
    res["const_sigma"], res["const_crps"] = const_s, const_crps
    learned = float(crps_gaussian(y, mu, scaled).mean())
    print(
        f"\n  learned per-protein sigma CRPS  {learned:.2f}"
        f"   vs best constant sigma {const_s:.2f} degC CRPS {const_crps:.2f}"
    )

    rho = float(spearmanr(sigma, err)[0])
    cv = float(np.std(scaled) / np.mean(scaled))
    res["sigma_error_rho"] = rho
    res["width_cv"] = cv
    q = np.quantile(sigma, [0.2, 0.4, 0.6, 0.8])
    quint = [float(err[np.digitize(sigma, q) == i].mean()) for i in range(5)]
    res["quintile_mae"] = quint
    print(f"\n  rank corr(sigma, |error|)   {rho:.3f}")
    print(f"  interval width CV           {cv:.2f}")
    print("  MAE by sigma quintile       " + ", ".join(f"{v:.2f}" for v in quint))
    return res


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, tm_mean, tm_std = train_stats()
    train_seqs = contaminated_train_seqs()
    kept_p, kept_f = kept_index_maps(train_seqs)
    base_all = torch.load(
        os.path.join(PROJECT_ROOT, "new_data/baseline_predictions.pt"),
        map_location="cpu",
        weights_only=False,
    )

    out = {}
    for loader, key, kept in [
        (load_protherm, "protherm", kept_p),
        (load_fireprot, "fireprot", kept_f),
    ]:
        tag, seqs, y, embs = loader(train_seqs)
        preds, _ = predict(seqs, embs, tm_mean, tm_std, device)
        mu, sigma = preds["v10 prior"]

        bl = {}
        for label, field in BASELINES.items():
            if field in base_all[key]:
                bl[label] = np.asarray(base_all[key][field], dtype=float)[kept]
        assert all(len(v) == len(y) for v in bl.values()), "baseline alignment failed"

        c, folds = fit_scale(y, mu, sigma, objective="crps")
        out[key] = {
            "n": int(len(y)),
            "scale_folds": folds,
            "table2": table2(tag, y, mu, sigma, c, bl),
            "per_bin": per_bin(tag, y, mu, bl),
            "calibration": calibration(tag, y, mu, sigma, c),
        }

        # Per-protein cache the figure scripts draw from, so no panel is ever plotted
        # from a literal or from a stale prediction set.
        cache = os.path.join(PROJECT_ROOT, f"paper/writeup/plots/_cache_{key}.npz")
        arrays = {
            "y_true": y,
            "y_pred": mu,
            "sigma": sigma,
            "sigma_scaled": c * sigma,
            "scale": c,
            "pred_StableProt": mu,
            "conf_StableProt": c * sigma,
            # Legacy keys the figure scripts still reference. The label says V9 but the
            # values are the shipped configuration, v9 T_m head on the v10 OGT prior.
            "pred_StableProt V9": mu,
            "conf_StableProt V9": c * sigma,
        }
        for label, pred in bl.items():
            arrays[f"pred_{label}"] = pred
            arrays[f"pred_{label.replace('-', '_')}"] = pred
        arrays.setdefault("pred_ThermoFormer", bl.get("ThermoFormer-TM", mu))
        np.savez(cache, **arrays)
        print(f"\nwrote {cache}")

    dest = os.path.join(PROJECT_ROOT, "paper/writeup/tables/refreshed_tm_numbers.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
