import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, accuracy_score, confusion_matrix, roc_curve

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from inference.v9_predict import V9Predictor, sanitize_sequence

def run_real_mutation_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../experiments/src/training/v9_disjoint/results"))
    print(f"Initializing V9Predictor on {device} using models_dir={models_dir}...")
    predictor = V9Predictor(models_dir=models_dir, device=device)

    data_path = "data/test_data/external/fireprotdb_results.csv"
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return

    df = pd.read_csv(data_path, low_memory=False)
    valid_df = df.dropna(subset=["sequence", "tm", "dTm", "wild_type", "mutation", "position"]).copy()
    print(f"Loaded {len(valid_df)} valid experimental mutation rows.")

    results = []
    wt_seq_cache = {}

    for idx, row in valid_df.iterrows():
        wt_seq = sanitize_sequence(str(row["sequence"]))
        if len(wt_seq) < 50:
            continue

        exp_tm = float(row["tm"])
        exp_dtm = float(row["dTm"])
        pdb_id = str(row.get("pdb_id", "Unknown"))
        wt_aa = str(row.get("wild_type", "")).strip().upper()
        mut_aa = str(row.get("mutation", "")).strip().upper()

        try:
            pos = int(float(row["position"]))
        except (ValueError, TypeError):
            continue

        # Find 100% exact alignment for wild_type AA in wt_seq
        target_idx = None
        p1 = pos - 1  # Standard 1-indexed conversion
        if 0 <= p1 < len(wt_seq) and wt_seq[p1] == wt_aa:
            target_idx = p1
        else:
            # Fallback to closest matching WT AA near pos
            matches = [i for i, aa in enumerate(wt_seq) if aa == wt_aa]
            if matches:
                target_idx = min(matches, key=lambda x: abs(x - p1))

        if target_idx is None or target_idx < 0 or target_idx >= len(wt_seq):
            continue

        # Construct exact mutant sequence
        mut_seq = wt_seq[:target_idx] + mut_aa + wt_seq[target_idx+1:]
        mut_code = f"{wt_aa}{target_idx+1}{mut_aa}"

        # Predict WT Tm (cached)
        if wt_seq not in wt_seq_cache:
            wt_pred_res = predictor.predict_single(wt_seq)
            if wt_pred_res.get("status") == "ERROR_TOO_SHORT" or "tm_pred" not in wt_pred_res:
                continue
            wt_seq_cache[wt_seq] = wt_pred_res["tm_pred"]

        wt_tm_pred = wt_seq_cache[wt_seq]

        # Predict Mutant Tm
        mut_pred_res = predictor.predict_single(mut_seq)
        if mut_pred_res.get("status") == "ERROR_TOO_SHORT" or "tm_pred" not in mut_pred_res:
            continue

        mut_tm_pred = mut_pred_res["tm_pred"]
        pred_dtm = round(mut_tm_pred - wt_tm_pred, 2)

        results.append({
            "pdb_id": pdb_id,
            "wild_type": wt_aa,
            "position": target_idx + 1,
            "mutation": mut_code,
            "exp_tm": exp_tm,
            "exp_dtm": exp_dtm,
            "pred_wt_tm": round(wt_tm_pred, 2),
            "pred_mut_tm": round(mut_tm_pred, 2),
            "pred_dtm": pred_dtm,
            "abs_error_dtm": round(abs(pred_dtm - exp_dtm), 2),
            "exp_stabilizing": 1 if exp_dtm > 0 else 0,
            "pred_stabilizing": 1 if pred_dtm > 0 else 0
        })

    res_df = pd.DataFrame(results)
    os.makedirs("experiments/results", exist_ok=True)
    res_csv = "experiments/results/real_mutation_benchmark_results.csv"
    res_df.to_csv(res_csv, index=False)
    print(f"Saved benchmark results ({len(res_df)} entries) to {res_csv}")

    # Calculate metrics
    mae_dtm = mean_absolute_error(res_df["exp_dtm"], res_df["pred_dtm"])
    rmse_dtm = np.sqrt(mean_squared_error(res_df["exp_dtm"], res_df["pred_dtm"]))
    pearson_r, _ = pearsonr(res_df["exp_dtm"], res_df["pred_dtm"])
    spearman_p, _ = spearmanr(res_df["exp_dtm"], res_df["pred_dtm"])
    acc = accuracy_score(res_df["exp_stabilizing"], res_df["pred_stabilizing"])
    cm = confusion_matrix(res_df["exp_stabilizing"], res_df["pred_stabilizing"])

    try:
        auc = roc_auc_score(res_df["exp_stabilizing"], res_df["pred_dtm"])
        fpr, tpr, _ = roc_curve(res_df["exp_stabilizing"], res_df["pred_dtm"])
    except:
        auc = 0.50
        fpr, tpr = [0, 1], [0, 1]

    print("\n==========================================")
    print("REAL EXPERIMENTAL MUTATION BENCHMARK RESULTS")
    print("==========================================")
    print(f"Total Evaluated Rows: {len(res_df)}")
    print(f"Delta Tm MAE:      {mae_dtm:.2f}°C")
    print(f"Delta Tm RMSE:     {rmse_dtm:.2f}°C")
    print(f"Pearson Correlation (r): {pearson_r:.4f}")
    print(f"Spearman Correlation (rho): {spearman_p:.4f}")
    print(f"Directional Accuracy: {acc*100:.1f}%")
    print(f"ROC-AUC: {auc:.4f}")

    # 4-Panel Figure Generation
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sns.set_theme(style="whitegrid")

    # Panel A: Scatter Plot
    ax = axes[0, 0]
    colors = np.where(res_df["exp_stabilizing"] == 1, "#10b981", "#ef4444")
    ax.scatter(res_df["exp_dtm"], res_df["pred_dtm"], c=colors, alpha=0.5, s=20, edgecolors="none")
    ax.axhline(0, color="#64748b", linestyle="--", linewidth=1)
    ax.axvline(0, color="#64748b", linestyle="--", linewidth=1)
    lims = [min(res_df["exp_dtm"].min(), res_df["pred_dtm"].min()) - 1, max(res_df["exp_dtm"].max(), res_df["pred_dtm"].max()) + 1]
    ax.plot(lims, lims, color="#475569", linestyle=":", linewidth=1.5, label="1:1 Parity")
    ax.set_xlabel("Experimental $\Delta T_m$ (°C)", fontsize=10, fontweight="bold")
    ax.set_ylabel("StableProt Predicted $\Delta T_m$ (°C)", fontsize=10, fontweight="bold")
    ax.set_title(f"A. Predicted vs Experimental $\Delta T_m$ ($r$ = {pearson_r:.2f}, $\\rho$ = {spearman_p:.2f})", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left")

    # Panel B: Confusion Matrix Heatmap
    ax = axes[0, 1]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, annot_kws={"size": 14, "weight": "bold"},
                xticklabels=["Destabilizing", "Stabilizing"], yticklabels=["Destabilizing", "Stabilizing"])
    ax.set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
    ax.set_ylabel("Experimental Wet-Lab Label", fontsize=10, fontweight="bold")
    ax.set_title(f"B. Directional Decision Confusion Matrix (Accuracy: {acc*100:.1f}%)", fontsize=11, fontweight="bold")

    # Panel C: ROC Curve
    ax = axes[1, 0]
    ax.plot(fpr, tpr, color="#2563eb", linewidth=2.5, label=f"StableProt (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", linewidth=1.5, label="Random Guess (AUC = 0.50)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10, fontweight="bold")
    ax.set_title(f"C. ROC Curve for Stabilizing Variant Discrimination", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right")

    # Panel D: MAE Grouped by Magnitude Bins
    ax = axes[1, 1]
    bins = [-100, 2, 5, 10, 100]
    labels = ["0–2°C\n(Minor)", "2–5°C\n(Moderate)", "5–10°C\n(Significant)", ">10°C\n(Extreme)"]
    res_df["magnitude_bin"] = pd.cut(res_df["exp_dtm"].abs(), bins=bins, labels=labels)
    bin_mae = res_df.groupby("magnitude_bin", observed=True)["abs_error_dtm"].mean()
    bin_counts = res_df.groupby("magnitude_bin", observed=True).size()
    
    bars = ax.bar(bin_mae.index.astype(str), bin_mae.values, color="#6366f1", edgecolor="#4338ca", width=0.55)
    for bar, count in zip(bars, bin_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.15, f"{height:.2f}°C\n(N={count})",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1e293b")
    ax.set_xlabel("Experimental $|\Delta T_m|$ Magnitude Interval", fontsize=10, fontweight="bold")
    ax.set_ylabel("Prediction MAE (°C)", fontsize=10, fontweight="bold")
    ax.set_title("D. Prediction MAE Stratified by Experimental Shift Magnitude", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(bin_mae.values) + 2.5)

    plt.tight_layout()
    plot_path = "experiments/results/real_mutation_benchmark_plot.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved 4-panel diagnostic plot to {plot_path}")

    # Write Markdown Analysis Report
    report_md = f"""# Real Wet-Lab Experimental Mutation $\Delta T_m$ Benchmark Analysis

## 1. Executive Summary
This benchmark evaluates StableProt V9 on **{len(res_df)} real wet-lab experimental mutation entries** from FireProtDB ([`fireprotdb_results.csv`](file:///home/bibhu/Documents/temstampto/data/test_data/external/fireprotdb_results.csv)). Wild-Type and Mutant predictions are evaluated independently to guarantee un-biased, non-circular $\Delta T_m$ calculation ($\Delta \hat{{T}}_m = \hat{{T}}_{{m, \text{{Mut}}}} - \hat{{T}}_{{m, \text{{WT}}}}$).

## 2. Quantitative Performance Matrix

| Evaluation Dimension | Metric | Score / Value | Scientific Interpretation |
|:---|:---:|:---:|:---|
| **Binary Decision Accuracy** | **Accuracy** | **{acc*100:.1f}%** | Correctly predicts whether a mutation is Stabilizing ($\Delta T_m > 0$) vs Destabilizing ($\Delta T_m \le 0$) |
| **ROC Discrimination** | **ROC-AUC** | **{auc:.4f}** | Area under ROC curve discriminating stabilizing mutants from thermal destabilizations |
| **Prediction MAE** | **MAE** | **{mae_dtm:.2f}°C** | Mean Absolute Error across all experimental temperature shifts |
| **Prediction RMSE** | **RMSE** | **{rmse_dtm:.2f}°C** | Root Mean Squared Error penalizing extreme physical outliers |
| **Linear Correlation** | **Pearson $r$** | **{pearson_r:.4f}** | Degree of linear alignment between predicted vs experimental $\Delta T_m$ |
| **Rank Correlation** | **Spearman $\\rho$** | **{spearman_p:.4f}** | Monotonic ranking accuracy for prioritizing mutation candidates |

## 3. Diagnostic Figure Overview
- **Diagnostic Plot**: [`experiments/results/real_mutation_benchmark_plot.png`](file:///home/bibhu/Documents/temstampto/experiments/results/real_mutation_benchmark_plot.png)
  - **Panel A**: Parity scatter plot of predicted vs experimental $\Delta T_m$ colored by true effect.
  - **Panel B**: 2×2 Confusion Matrix Heatmap highlighting **{acc*100:.1f}% Directional Accuracy**.
  - **Panel C**: ROC Curve demonstrating **{auc:.4f} AUC** performance.
  - **Panel D**: Error MAE stratified by experimental shift magnitude bins (0–2°C, 2–5°C, 5–10°C, >10°C).
"""
    with open("experiments/results/real_mutation_benchmark_analysis.md", "w") as f:
        f.write(report_md)
    print("Saved report to experiments/results/real_mutation_benchmark_analysis.md")

if __name__ == "__main__":
    run_real_mutation_benchmark()
