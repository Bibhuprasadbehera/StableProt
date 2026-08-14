#!/usr/bin/env python3
"""C.4a: does swapping the OGT prior source change T_m?

The T_m head consumes a predicted OGT prior, so adopting the v10 OGT head as the reported OGT
model also changes every T_m prediction. This scores the same v9 T_m ensemble on ProThermDB and
FireProtDB under both prior sources and reports a paired bootstrap on the difference.

If T_m holds within noise, one OGT head can serve both roles. If it degrades, the v9 head has to
stay as the frozen internal prior source and that has to be disclosed.

Run from the repo root inside the `stableprot_v2` conda environment.
"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from scipy.stats import pearsonr, spearmanr, norm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
V9_DIR = os.path.join(PROJECT_ROOT, "experiments/src/training/v9_disjoint")
V10_DIR = os.path.join(PROJECT_ROOT, "experiments/src/training/v10")
SEEDS = range(1, 6)
N_BOOT = 4000
RNG = np.random.default_rng(0)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def crps_gaussian(y, mu, sigma):
    sigma = np.maximum(sigma, 1e-6)
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def fit_sigma_scale(y, mu, sigma):
    """Two-fold cross-fitted CRPS-minimising scale, so no point sets its own scale."""
    n = len(y)
    order = RNG.permutation(n)
    folds = [order[: n // 2], order[n // 2 :]]
    grid = np.linspace(0.2, 8.0, 400)
    scaled = np.empty(n)
    for held, fit in [(0, 1), (1, 0)]:
        f = folds[fit]
        scores = [crps_gaussian(y[f], mu[f], c * sigma[f]).mean() for c in grid]
        c = grid[int(np.argmin(scores))]
        scaled[folds[held]] = c
    return scaled


def train_stats():
    v9_train = load_module("train_v9", os.path.join(V9_DIR, "train.py"))
    data = torch.load(
        os.path.join(PROJECT_ROOT, "data/embeddings/saprot_tm_struct_embeddings.pt"),
        map_location="cpu",
        weights_only=False,
    )
    _, _, lbl, _, _ = v9_train.sanitize_data(data["train_tm"], is_tm=True)
    return v9_train, lbl.mean().item(), lbl.std().item()


def contaminated_train_seqs():
    seqs = set()
    for p in [
        "data/embeddings/prepared_data_v7_saprot1.3b_seqonly.pt",
        "data/embeddings/saprot_tm_struct_embeddings.pt",
    ]:
        full = os.path.join(PROJECT_ROOT, p)
        if os.path.exists(full):
            d = torch.load(full, map_location="cpu", weights_only=False)
            if "train_tm" in d and "sequences" in d["train_tm"]:
                seqs.update(str(s).upper() for s in d["train_tm"]["sequences"])
    return seqs


def load_protherm(train_seqs):
    df = pd.read_csv(os.path.join(PROJECT_ROOT, "new_data/prothermdb_validation.csv"))
    tm_by_uid = {
        str(r["UniProt_ID"]): float(r["Tm"]) for _, r in df.iterrows() if not np.isnan(r["Tm"])
    }
    seqs, y, kept, idx = [], [], [], 0
    for rec in SeqIO.parse(os.path.join(PROJECT_ROOT, "new_data/prothermdb_validation.fasta"), "fasta"):
        s = str(rec.seq)
        uid = rec.id.split("|")[0]
        if uid in tm_by_uid:
            if s.upper() not in train_seqs:
                seqs.append(s)
                y.append(tm_by_uid[uid])
                kept.append(idx)
            idx += 1

    base = torch.load(
        os.path.join(PROJECT_ROOT, "data/embeddings/saprot_1.3b/protherm_embeddings.pt"),
        map_location="cpu",
        weights_only=False,
    ).float()[kept]
    struct_path = os.path.join(PROJECT_ROOT, "data/embeddings/protherm_v8_struct_embeddings.pt")
    if os.path.exists(struct_path):
        d = torch.load(struct_path, map_location="cpu", weights_only=False)
        base = torch.stack([d.get(s, base[i]) for i, s in enumerate(seqs)], dim=0)
    return "ProThermDB", seqs, np.array(y), base


def load_fireprot(train_seqs):
    d_saprot = torch.load(
        os.path.join(PROJECT_ROOT, "data/test_data/fireprot_holdout_saprot.pt"),
        map_location="cpu",
        weights_only=False,
    )
    d_prott5 = torch.load(
        os.path.join(PROJECT_ROOT, "experiments/src/data/fireprot_holdout_prott5.pt"),
        map_location="cpu",
        weights_only=False,
    )
    seqs_all = [str(s) for s in d_saprot["sequences"]]
    kept = [i for i, s in enumerate(seqs_all) if s.upper() not in train_seqs]
    temps = d_prott5["temperatures"]
    y = (temps.numpy() if hasattr(temps, "numpy") else np.array(temps))[kept]
    seqs = [seqs_all[i] for i in kept]

    base = d_saprot["embeddings_saprot"].float()[kept]
    struct_path = os.path.join(PROJECT_ROOT, "data/embeddings/fireprot_v8_struct_embeddings.pt")
    if os.path.exists(struct_path):
        d = torch.load(struct_path, map_location="cpu", weights_only=False)
        base = torch.stack([d.get(s, base[i]) for i, s in enumerate(seqs)], dim=0)
    return "FireProtDB", seqs, y, base


def predict(seqs, embs, tm_mean, tm_std, device):
    """Returns {prior_source: (mu, sigma)} for the same v9 T_m ensemble."""
    v9_train = load_module("train_v9m", os.path.join(V9_DIR, "train.py"))
    v10_train = load_module("train_v10m", os.path.join(V10_DIR, "train.py"))
    norms = torch.load(
        os.path.join(V9_DIR, "results/normalization_stats.pt"), map_location="cpu", weights_only=False
    )
    o_mean, o_std = norms["ogt_mean"], norms["ogt_std"]

    out = {"v9 prior": [[], []], "v10 prior": [[], []]}
    priors = {}
    for s in SEEDS:
        tm_p = os.path.join(V9_DIR, f"results/seed{s}/model_tm.pt")
        o9_p = os.path.join(V9_DIR, f"results/seed{s}/model_ogt.pt")
        o10_p = os.path.join(V10_DIR, f"results/seed{s}/model_ogt.pt")
        if not all(os.path.exists(p) for p in (tm_p, o9_p, o10_p)):
            print(f"  seed{s}: missing checkpoint, skipped")
            continue

        m_tm = v9_train.MultiHeadSaProtV8().to(device)
        m_tm.load_state_dict(torch.load(tm_p, map_location=device, weights_only=False))
        m_o9 = v9_train.MultiHeadSaProtV8().to(device)
        m_o9.load_state_dict(torch.load(o9_p, map_location=device, weights_only=False))
        m_o10 = v10_train.MultiHeadSaProtV8(ogt_heteroscedastic=True).to(device)
        m_o10.load_state_dict(torch.load(o10_p, map_location=device, weights_only=False))
        for m in (m_tm, m_o9, m_o10):
            m.eval()

        with torch.no_grad():
            emb_o, aux_o = v9_train.enrich_inputs(embs, seqs, tmhmm_flags=None, ogt_priors=None)
            emb_o, aux_o = emb_o.to(device), aux_o.to(device)

            p9 = m_o9(emb_o, aux_o, head="ogt").cpu() * o_std + o_mean
            p10 = m_o10(emb_o, aux_o, head="ogt")[0].cpu() * o_std + o_mean
            priors.setdefault("v9 prior", []).append(p9.numpy())
            priors.setdefault("v10 prior", []).append(p10.numpy())

            for name, prior in [("v9 prior", p9), ("v10 prior", p10)]:
                emb_t, aux_t = v9_train.enrich_inputs(
                    embs, seqs, tmhmm_flags=None, ogt_priors=prior.numpy()
                )
                z_mu, z_var = m_tm(emb_t.to(device), aux_t.to(device), head="tm")
                out[name][0].append((z_mu.cpu() * tm_std + tm_mean).numpy())
                out[name][1].append((z_var.cpu() * tm_std**2).numpy())

    final = {}
    for name, (mus, vars_) in out.items():
        mus, vars_ = np.stack(mus), np.stack(vars_)
        w = 1.0 / (vars_ + 1e-6)
        mu = (mus * w).sum(0) / w.sum(0)
        # Law of total variance, not the standard error of the ensemble mean.
        sigma = np.sqrt(vars_.mean(0) + ((mus - mu) ** 2).mean(0))
        final[name] = (mu, sigma)
    return final, {k: np.mean(v, axis=0) for k, v in priors.items()}


def report(tag, y, preds, prior_vals):
    print(f"\n{'='*74}\n{tag}   n = {len(y)}\n{'='*74}")

    p9, p10 = prior_vals["v9 prior"], prior_vals["v10 prior"]
    print(
        f"OGT prior itself:  v9 mean {p9.mean():6.2f} sd {p9.std():5.2f}   |   "
        f"v10 mean {p10.mean():6.2f} sd {p10.std():5.2f}   |   "
        f"mean |shift| {np.abs(p10 - p9).mean():5.2f} degC   r {pearsonr(p9, p10)[0]:.3f}"
    )

    rows = {}
    for name, (mu, sigma) in preds.items():
        c = fit_sigma_scale(y, mu, sigma)
        rows[name] = dict(
            mae=np.abs(y - mu),
            crps=crps_gaussian(y, mu, c * sigma),
            r=pearsonr(y, mu)[0],
            rho=spearmanr(y, mu)[0],
            c=float(np.median(c)),
        )

    print(f"\n{'prior':<12}{'MAE':>9}{'CRPS':>9}{'r':>9}{'rho':>9}{'c':>7}")
    for name, m in rows.items():
        print(
            f"{name:<12}{m['mae'].mean():9.3f}{m['crps'].mean():9.3f}"
            f"{m['r']:9.3f}{m['rho']:9.3f}{m['c']:7.2f}"
        )

    a, b = rows["v9 prior"], rows["v10 prior"]
    n = len(y)
    print(f"\nPaired bootstrap, {N_BOOT} resamples, v10 minus v9 (negative favours the v10 prior):")
    for metric in ("mae", "crps"):
        d = b[metric] - a[metric]
        boot = np.array([d[RNG.integers(0, n, n)].mean() for _ in range(N_BOOT)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        verdict = "tie" if lo <= 0 <= hi else ("v10 better" if hi < 0 else "v9 better")
        print(f"  d{metric.upper():<5} {d.mean():+7.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]   {verdict}")
    return rows


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    _, tm_mean, tm_std = train_stats()
    print(f"T_m normalisation: mean {tm_mean:.3f}, std {tm_std:.3f}")
    train_seqs = contaminated_train_seqs()
    print(f"training sequences held for decontamination: {len(train_seqs)}")

    for loader in (load_protherm, load_fireprot):
        tag, seqs, y, embs = loader(train_seqs)
        preds, priors = predict(seqs, embs, tm_mean, tm_std, device)
        report(tag, y, preds, priors)

    print(
        "\nDecision rule (C.4a): if both dMAE and dCRPS intervals contain zero on both benchmarks,"
        "\nship a single v10 OGT head serving the reported OGT task and the T_m prior."
    )


if __name__ == "__main__":
    main()
