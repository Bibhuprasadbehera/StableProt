import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from inference.v9_predict import V9Predictor, sanitize_sequence, classify_tm_tier
from inference.v7_predict import get_saprot_embedding

def run_experiment():
    models_dir = os.path.join(root_dir, "experiments/src/training/v9_disjoint/results")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing V9Predictor on {device}...")
    predictor = V9Predictor(models_dir=models_dir, device=device)

    # Load candidates from FLIP Meltome clean dataset
    flip_csv = os.path.join(root_dir, "data/flip_meltome/flip_clean.csv")
    df_flip = pd.read_csv(flip_csv)

    target_ids = ["Sequence7558", "Sequence4486", "Sequence1675", "Sequence3328", "Sequence13050"]
    selected_proteins = df_flip[df_flip["seqid"].isin(target_ids)].copy()

    print(f"Selected {len(selected_proteins)} test proteins:")
    for _, row in selected_proteins.iterrows():
        print(f"  {row['seqid']}: Exp Tm = {row['label']:.2f}°C, Length = {len(row['sequence'])} aa")

    records = []

    for idx, row in selected_proteins.iterrows():
        prot_id = row["seqid"]
        exp_tm = float(row["label"])
        wt_seq = sanitize_sequence(row["sequence"])
        tier = classify_tm_tier(exp_tm)

        print(f"\n==========================================")
        print(f"Processing {prot_id} (Exp Tm: {exp_tm:.2f}°C, Tier: {tier}, Len: {len(wt_seq)})")
        print(f"==========================================")

        # Baseline WT SaProt embedding
        wt_emb = get_saprot_embedding(predictor.embed_model, predictor.tokenizer, wt_seq, device=device)

        # Round 0 (WT baseline)
        wt_pred = predictor.predict_single(wt_seq)
        wt_tm_pred = wt_pred["tm_pred"]
        wt_abs_err = abs(wt_tm_pred - exp_tm)

        records.append({
            "protein_id": prot_id,
            "exp_tm": exp_tm,
            "tier": tier,
            "round": 0,
            "n_mutations": 0,
            "mutation_applied": "WT",
            "predicted_tm": wt_tm_pred,
            "predicted_conf": wt_pred["tm_conf"],
            "absolute_error": round(wt_abs_err, 2),
            "seq_identity_to_wt": 1.0000,
            "embedding_l2_dist": 0.0000,
            "cosine_similarity": 1.0000
        })

        current_seq = wt_seq
        n_rounds = 15

        for r in range(1, n_rounds + 1):
            print(f"  Round {r}/{n_rounds} for {prot_id}...")
            # Run mutagenesis scan across all positions (blind mode)
            scan_df = predictor.predict_mutants(current_seq, positions=None)
            if scan_df.empty:
                print(f"  No valid scan results in round {r}, breaking loop.")
                break

            # Select top stabilizing candidate
            candidates = scan_df[scan_df["delta_tm"] > 0]
            if candidates.empty:
                candidates = scan_df  # Fallback to top candidate even if delta_tm <= 0

            best = candidates.iloc[0]
            pos_0 = int(best["position"]) - 1
            mut_label = f"{best['wt_aa']}{best['position']}{best['mut_aa']}"
            current_seq = current_seq[:pos_0] + best["mut_aa"] + current_seq[pos_0+1:]

            # Evaluate updated mutant sequence
            mut_pred = predictor.predict_single(current_seq)
            mut_tm_pred = mut_pred["tm_pred"]
            abs_err = abs(mut_tm_pred - exp_tm)
            seq_ident = sum(a == b for a, b in zip(wt_seq, current_seq)) / len(wt_seq)

            # Compute mutant embedding metrics
            mut_emb = get_saprot_embedding(predictor.embed_model, predictor.tokenizer, current_seq, device=device)
            l2_dist = torch.norm(wt_emb.float() - mut_emb.float(), p=2).item()
            cos_sim = torch.nn.functional.cosine_similarity(wt_emb.float(), mut_emb.float(), dim=-1).mean().item()

            records.append({
                "protein_id": prot_id,
                "exp_tm": exp_tm,
                "tier": tier,
                "round": r,
                "n_mutations": r,
                "mutation_applied": mut_label,
                "predicted_tm": mut_tm_pred,
                "predicted_conf": mut_pred["tm_conf"],
                "absolute_error": round(abs_err, 2),
                "seq_identity_to_wt": round(seq_ident, 4),
                "embedding_l2_dist": round(l2_dist, 4),
                "cosine_similarity": round(cos_sim, 4)
            })

            print(f"    Mut: {mut_label} -> Pred Tm: {mut_tm_pred:.2f}°C (Err: {abs_err:.2f}°C), Ident: {seq_ident:.2%}, L2 Dist: {l2_dist:.2f}")

    # Save results DataFrame
    df_results = pd.DataFrame(records)
    out_csv = os.path.join(root_dir, "experiments/results/iterative_error_accumulation.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"\nSaved experiment results to {out_csv}")

    # Plot 2x2 grid
    plot_results(df_results)
    generate_markdown_report(df_results)

def plot_results(df):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    color_map = {"LOW": "#3b82f6", "MEDIUM": "#f59e0b", "HIGH": "#ef4444"}

    # Panel A: Round vs Absolute Error
    ax1 = axes[0, 0]
    for prot_id, grp in df.groupby("protein_id"):
        tier = grp["tier"].iloc[0]
        exp_tm = grp["exp_tm"].iloc[0]
        ax1.plot(grp["round"], grp["absolute_error"], marker='o', linewidth=2, label=f"{prot_id} (Exp: {exp_tm:.1f}°C)", color=color_map[tier])
    ax1.set_xlabel("Iterative Round", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Absolute Error vs Ground Truth Tm (°C)", fontsize=11, fontweight='bold')
    ax1.set_title("(A) Error Accumulation Across Iterative Rounds", fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper left')

    # Panel B: Round vs Sequence Identity to WT
    ax2 = axes[0, 1]
    for prot_id, grp in df.groupby("protein_id"):
        tier = grp["tier"].iloc[0]
        ax2.plot(grp["round"], grp["seq_identity_to_wt"] * 100, marker='s', linewidth=2, label=prot_id, color=color_map[tier])
    ax2.set_xlabel("Iterative Round", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Sequence Identity to WT (%)", fontsize=11, fontweight='bold')
    ax2.set_title("(B) Sequence Identity Decay per Round", fontsize=12, fontweight='bold')

    # Panel C: Embedding L2 Distance vs Absolute Error
    ax3 = axes[1, 0]
    for prot_id, grp in df.groupby("protein_id"):
        tier = grp["tier"].iloc[0]
        ax3.scatter(grp["embedding_l2_dist"], grp["absolute_error"], label=prot_id, color=color_map[tier], s=40, alpha=0.8)
        ax3.plot(grp["embedding_l2_dist"], grp["absolute_error"], color=color_map[tier], linestyle='--', alpha=0.5)
    ax3.set_xlabel("SaProt Embedding L2 Distance to WT", fontsize=11, fontweight='bold')
    ax3.set_ylabel("Absolute Error vs Ground Truth Tm (°C)", fontsize=11, fontweight='bold')
    ax3.set_title("(C) Error Trajectory vs Embedding Representation Shift", fontsize=12, fontweight='bold')

    # Panel D: Round vs Predicted Tm (with Ground Truth lines)
    ax4 = axes[1, 1]
    for prot_id, grp in df.groupby("protein_id"):
        tier = grp["tier"].iloc[0]
        exp_tm = grp["exp_tm"].iloc[0]
        ax4.plot(grp["round"], grp["predicted_tm"], marker='^', linewidth=2, label=f"{prot_id} Pred", color=color_map[tier])
        ax4.axhline(exp_tm, color=color_map[tier], linestyle=':', alpha=0.7, label=f"{prot_id} Exp ({exp_tm:.1f}°C)")
    ax4.set_xlabel("Iterative Round", fontsize=11, fontweight='bold')
    ax4.set_ylabel("Predicted Melting Temperature (°C)", fontsize=11, fontweight='bold')
    ax4.set_title("(D) Predicted Tm Trajectories vs Ground Truth", fontsize=12, fontweight='bold')
    ax4.legend(fontsize=8, loc='upper left', ncol=2)

    plt.tight_layout()
    out_png = os.path.join(root_dir, "experiments/results/iterative_error_accumulation_plot.png")
    plt.savefig(out_png)
    plt.close()
    print(f"Saved plot to {out_png}")

def generate_markdown_report(df):
    report_path = os.path.join(root_dir, "experiments/results/iterative_error_analysis.md")
    
    # Calculate statistics across rounds
    mean_err_r0 = df[df["round"] == 0]["absolute_error"].mean()
    mean_err_r5 = df[df["round"] == 5]["absolute_error"].mean()
    mean_err_r10 = df[df["round"] == 10]["absolute_error"].mean()
    mean_err_r15 = df[df["round"] == 15]["absolute_error"].mean()
    
    content = f"""# Empirical Analysis: Iterative Mutation Error Accumulation & Representation Shift

## Executive Summary

This experiment empirically evaluates how prediction error evolves across **15 consecutive rounds of iterative directed evolution** on five test proteins spanning LOW (<45°C), MEDIUM (45–65°C), and HIGH (≥65°C) thermostability tiers.

### Mean Error Progression
- **Round 0 (WT Baseline)**: Mean MAE = **{mean_err_r0:.2f}°C**
- **Round 5 (5 Mutations)**: Mean MAE = **{mean_err_r5:.2f}°C**
- **Round 10 (10 Mutations)**: Mean MAE = **{mean_err_r10:.2f}°C**
- **Round 15 (15 Mutations)**: Mean MAE = **{mean_err_r15:.2f}°C**

---

## Detailed Per-Protein Progression Table

| Protein ID | Exp Tm (°C) | Tier | WT Error (R0) | R5 Error | R10 Error | R15 Error | Final Ident (%) | Max L2 Dist |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for prot_id, grp in df.groupby("protein_id"):
        r0_err = grp[grp["round"] == 0]["absolute_error"].values[0]
        r5_err = grp[grp["round"] == 5]["absolute_error"].values[0] if 5 in grp["round"].values else N/A
        r10_err = grp[grp["round"] == 10]["absolute_error"].values[0] if 10 in grp["round"].values else N/A
        r15_err = grp[grp["round"] == 15]["absolute_error"].values[0] if 15 in grp["round"].values else N/A
        exp_tm = grp["exp_tm"].iloc[0]
        tier = grp["tier"].iloc[0]
        final_ident = grp["seq_identity_to_wt"].iloc[-1] * 100
        max_l2 = grp["embedding_l2_dist"].max()
        content += f"| **{prot_id}** | {exp_tm:.1f} | {tier} | {r0_err:.2f}°C | {r5_err:.2f}°C | {r10_err:.2f}°C | {r15_err:.2f}°C | {final_ident:.1f}% | {max_l2:.2f} |\n"

    content += """
---

## Key Biophysical Insights

1. **Error Behavior**: Single-point mutations induce incremental prediction shift; cumulative error trajectory reveals whether model error accumulates monotonically or plateaus as representation shifts.
2. **Representation Distance**: SaProt 3Di embedding L2 distance correlates directly with structural token divergence, measuring the extent of representation shift across multi-mutation stacking.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved markdown report to {report_path}")

if __name__ == "__main__":
    run_experiment()
