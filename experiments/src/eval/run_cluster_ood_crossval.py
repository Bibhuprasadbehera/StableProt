#!/usr/bin/env python3
"""
Cluster-Based Out-of-Distribution (OOD) Family Generalization (`Claim 2`)

Clusters evaluation proteins (`val_tm + test_tm + protherm`) at 30% sequence identity
using MMseqs2 (`easy-cluster --min-seq-id 0.3 -c 0.8`), and evaluates StableProt V8
across the largest distinct family clusters to verify OOD generalization across unseen families.

Outputs:
  - `cluster_ood_summary.csv`
  - `cluster_ood_generalization.png`
  - `cluster_ood_generalization.json` (Universal JSON compliance)
"""

import os
import sys
import json
import subprocess
import tempfile
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
PROJECT_ROOT = os.path.dirname(EXPERIMENTS_DIR)
VERSION = os.environ.get("STABLEPROT_VERSION", "v8_disjoint")
sys.path.append(os.path.join(EXPERIMENTS_DIR, f"src/training/{VERSION}"))

from train import MultiHeadSaProtV8, enrich_inputs

OUT_DIR = os.path.join(PROJECT_ROOT, "paper/writeup/plots")
VAL_SUITE_DIR = os.path.join(EXPERIMENTS_DIR, "new_data/validation_suite")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(VAL_SUITE_DIR, exist_ok=True)

def run_mmseqs_clustering(fasta_path, tmp_dir):
    """Run mmseqs easy-cluster at 30% identity (`--min-seq-id 0.3 -c 0.8`)."""
    cluster_prefix = os.path.join(tmp_dir, "clusters_30pct")
    tmp_scratch = os.path.join(tmp_dir, "scratch")
    os.makedirs(tmp_scratch, exist_ok=True)
    
    mmseqs_bin = os.path.join(os.path.dirname(sys.executable), "mmseqs")
    if not os.path.exists(mmseqs_bin):
        mmseqs_bin = "mmseqs"  # fallback to PATH
    cmd = [
        mmseqs_bin, "easy-cluster", fasta_path, cluster_prefix, tmp_scratch,
        "--min-seq-id", "0.3", "-c", "0.8", "--cov-mode", "0", "-v", "1"
    ]
    print(f"Running MMseqs2 30% clustering: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    tsv_path = f"{cluster_prefix}_cluster.tsv"
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"MMseqs cluster tsv not found at {tsv_path}")
        
    return tsv_path

def load_v9_model(device):
    import inspect
    VERSION = os.environ.get("STABLEPROT_VERSION", "v8_disjoint")
    model_dir = os.path.join(EXPERIMENTS_DIR, f"src/training/{VERSION}/results/seed1")
    ckpt_path = os.path.join(model_dir, "model_tm.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    from config import CONFIG
    sig = inspect.signature(MultiHeadSaProtV8.__init__)
    model_kwargs = {'proj_dim': CONFIG.get('proj_dim', 64)}
    if 'use_residuals' in sig.parameters:
        model_kwargs['use_residuals'] = CONFIG.get('use_residuals', True)
    model = MultiHeadSaProtV8(**model_kwargs).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()
    return model

def main():
    print("=====================================================================================")
    print("  CLUSTER-BASED OOD FAMILY GENERALIZATION BENCHMARK (`Claim 2`)")
    print("=====================================================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Gather all evaluation sequences and embeddings (`val_tm + test_tm + protherm`)
    data_path = os.path.join(PROJECT_ROOT, "data/embeddings/saprot_tm_struct_embeddings.pt")
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    
    val_tm = data['val_tm']
    test_tm = data['test_tm']
    
    y_val = val_tm['tm_consensus'].numpy() if isinstance(val_tm['tm_consensus'], torch.Tensor) else np.array(val_tm['tm_consensus'])
    y_test = test_tm['tm_consensus'].numpy() if isinstance(test_tm['tm_consensus'], torch.Tensor) else np.array(test_tm['tm_consensus'])
    
    n_val = len(y_val)
    n_test = len(y_test)
    
    embs_list = [val_tm['embeddings'][:n_val].cpu(), test_tm['embeddings'][:n_test].cpu()]
    seqs_list = val_tm['sequences'][:n_val] + test_tm['sequences'][:n_test]
    y_list = [y_val, y_test]
    
    # Include ProTherm
    protherm_emb_p = os.path.join(PROJECT_ROOT, "data/embeddings/protherm_v8_struct_embeddings.pt")
    if os.path.exists(protherm_emb_p):
        p_data = torch.load(protherm_emb_p, map_location='cpu', weights_only=False)
        protherm_csv = os.path.join(PROJECT_ROOT, 'new_data/prothermdb_validation.csv')
        if os.path.exists(protherm_csv):
            df_p = pd.read_csv(protherm_csv)
            p_dict = {str(row['UniProt_ID']): float(row['Tm']) for _, row in df_p.iterrows() if not np.isnan(row['Tm'])}
            
            protherm_fasta = os.path.join(PROJECT_ROOT, 'new_data/prothermdb_validation.fasta')
            for record in SeqIO.parse(protherm_fasta, 'fasta'):
                seq = str(record.seq)
                uid = record.id.split('|')[0]
                if uid in p_dict and seq in p_data:
                    embs_list.append(p_data[seq].cpu().unsqueeze(0))
                    seqs_list.append(seq)
                    y_list.append(np.array([p_dict[uid]]))
                    
    embs_all = torch.cat(embs_list, dim=0)
    y_all = np.concatenate(y_list)
    print(f"Total Evaluation Samples: {len(y_all):,}")
    
    # 2. Export FASTA for MMseqs2 clustering
    with tempfile.TemporaryDirectory() as tmp_dir:
        fasta_path = os.path.join(tmp_dir, "eval_sequences.fasta")
        records = [SeqRecord(Seq(seq), id=f"seq_{i}", description="") for i, seq in enumerate(seqs_list)]
        SeqIO.write(records, fasta_path, "fasta")
        
        tsv_path = run_mmseqs_clustering(fasta_path, tmp_dir)
        df_clusters = pd.read_csv(tsv_path, sep="\t", header=None, names=["rep", "member"])
        
    # Map each member ID back to index
    member_indices = {row['member']: int(row['member'].split('_')[1]) for _, row in df_clusters.iterrows()}
    
    # Group by cluster representative (`rep`)
    cluster_groups = df_clusters.groupby('rep')['member'].apply(list).to_dict()
    cluster_sizes = {rep: len(members) for rep, members in cluster_groups.items()}
    
    # Sort clusters by size descending
    sorted_reps = sorted(cluster_sizes.keys(), key=lambda r: cluster_sizes[r], reverse=True)
    print(f"Total distinct 30% sequence clusters: {len(sorted_reps):,}")
    print(f"Top 10 cluster sizes: {[cluster_sizes[r] for r in sorted_reps[:10]]}")
    
    # 3. Evaluate V9 inference across all samples
    model = load_v9_model(device)
    emb_t, aux_t = enrich_inputs(embs_all, seqs_list, tmhmm_flags=None, ogt_priors=[50.0]*len(seqs_list))
    
    VERSION = os.environ.get("STABLEPROT_VERSION", "v8_disjoint")
    stats_path = os.path.join(EXPERIMENTS_DIR, f"src/training/{VERSION}/results/normalization_stats.pt")
    norms = torch.load(stats_path, map_location='cpu', weights_only=False) if os.path.exists(stats_path) else {'tm_mean': 56.4, 'tm_std': 13.2}
    
    preds_list = []
    with torch.no_grad():
        for i in range(0, len(emb_t), 512):
            mu_norm, _ = model(emb_t[i:i+512].to(device), aux_t[i:i+512].to(device), head='tm')
            preds_list.append(mu_norm.cpu().numpy() * norms['tm_std'] + norms['tm_mean'])
    preds_all = np.concatenate(preds_list)
    
    overall_mae = np.mean(np.abs(y_all - preds_all))
    print(f"Overall Baseline MAE across all clusters: {overall_mae:.4f}°C")
    
    # 4. Analyze top 8 largest family clusters
    top_clusters = sorted_reps[:8]
    cluster_results = []
    
    for idx, rep in enumerate(top_clusters, 1):
        members = cluster_groups[rep]
        indices = [member_indices[m] for m in members if m in member_indices]
        
        y_c = y_all[indices]
        p_c = preds_all[indices]
        
        mae_c = np.mean(np.abs(y_c - p_c))
        rmse_c = np.sqrt(np.mean((y_c - p_c)**2))
        tm_mean_c = np.mean(y_c)
        len_mean_c = np.mean([len(seqs_list[i]) for i in indices])
        
        cluster_results.append({
            "Cluster_Rank": f"Family Cluster #{idx}",
            "Rep_ID": rep,
            "Sample_Count": len(indices),
            "Mean_Tm_Target": round(float(tm_mean_c), 2),
            "Mean_Seq_Length": round(float(len_mean_c), 1),
            "V9_MAE": round(float(mae_c), 4),
            "V9_RMSE": round(float(rmse_c), 4)
        })
        print(f"  Family Cluster #{idx:2d} (N={len(indices):4d}) | Mean Tm: {tm_mean_c:5.1f}°C | MAE: {mae_c:.2f}°C")
        
    df_out = pd.DataFrame(cluster_results)
    csv_out = os.path.join(VAL_SUITE_DIR, "cluster_ood_summary.csv")
    df_out.to_csv(csv_out, index=False)
    print(f"\nSaved cluster OOD summary to: {csv_out}")
    
    # Save Universal JSON coordinates
    json_out = os.path.join(OUT_DIR, "cluster_ood_generalization.json")
    json_data = {
        "overall_baseline_mae": float(overall_mae),
        "clusters_30pct_identity": cluster_results
    }
    with open(json_out, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved JSON plot data: {json_out}")
    
    # 5. Plot bar diagram of Family-Level OOD Generalization
    sns.set_context("paper", font_scale=1.2)
    plt.figure(figsize=(9.5, 5.5))
    
    colors = ['#10b981' if r['V9_MAE'] <= overall_mae + 1.5 else '#f59e0b' for r in cluster_results]
    bars = plt.bar([r['Cluster_Rank'] for r in cluster_results], [r['V9_MAE'] for r in cluster_results], color=colors, alpha=0.85, edgecolor='black', linewidth=1)
    
    plt.axhline(y=overall_mae, color='#b91c1c', linestyle='--', linewidth=2.5, label=f'Overall Baseline MAE ({overall_mae:.2f}°C)')
    
    for bar, r in zip(bars, cluster_results):
        plt.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 0.2,
                 f"{r['V9_MAE']:.2f}°C\n(N={r['Sample_Count']})", ha='center', va='bottom', fontsize=9, fontweight='bold')
                 
    plt.ylabel("Prediction MAE (°C)")
    plt.title("Family-Level OOD Generalization across MMseqs2 30% Sequence Clusters (`Claim 2`)")
    plt.xticks(rotation=25, ha='right')
    plt.ylim(0, max([r['V9_MAE'] for r in cluster_results]) * 1.25)
    plt.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    
    p_out = os.path.join(OUT_DIR, "cluster_ood_generalization.png")
    plt.savefig(p_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cluster generalization plot: {p_out}")

if __name__ == "__main__":
    main()
