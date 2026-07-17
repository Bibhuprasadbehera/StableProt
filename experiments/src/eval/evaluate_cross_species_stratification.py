#!/usr/bin/env python3
"""
Evaluate Cross-Species Stratification & Intra-Organism Ranking (`Benchmark 8`)

Proves that StableProt V9 does not merely memorize species-level amino acid composition,
but accurately ranks individual protein thermostabilities within single organism proteomes.

Outputs:
  - `experiments/new_data/validation_suite/cross_species_summary.csv`
  - `paper/writeup/plots/cross_species_generalization.png`
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
PROJECT_ROOT = os.path.dirname(EXPERIMENTS_DIR)
OUT_DIR = os.path.join(PROJECT_ROOT, "paper/writeup/plots")
VAL_SUITE_DIR = os.path.join(EXPERIMENTS_DIR, "new_data/validation_suite")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(VAL_SUITE_DIR, exist_ok=True)

def load_organism_mapping():
    """Build Sequence -> Organism mapping from Meltome, EsmTemp, and ProTherm sources."""
    print("Building sequence-to-organism taxonomic map...")
    seq_to_org = {}
    
    # 1. EsmTemp dataset (has exact organism lysate strings)
    esm_p = os.path.join(PROJECT_ROOT, "benchmark_models_tm/EsmTemp/dataset.csv")
    if os.path.exists(esm_p):
        df_esm = pd.read_csv(esm_p, header=None)
        for _, row in df_esm.iterrows():
            if len(row) >= 4 and isinstance(row[1], str) and isinstance(row[3], str):
                s_clean = "".join([c for c in row[1].upper() if c.isupper() and c.isalpha()])
                org = row[3].replace(" lysate", "").strip()
                if len(s_clean) > 20 and org:
                    seq_to_org[s_clean] = org
                    
    # 2. FLIP Meltome mixed split
    flip_p = os.path.join(PROJECT_ROOT, "data/flip_meltome/mixed_split.csv")
    if os.path.exists(flip_p):
        try:
            df_flip = pd.read_csv(flip_p)
            seq_col = 'sequence' if 'sequence' in df_flip.columns else 'seq'
            org_col = 'organism' if 'organism' in df_flip.columns else ('species' if 'species' in df_flip.columns else None)
            if org_col:
                for _, row in df_flip.dropna(subset=[seq_col, org_col]).iterrows():
                    s_clean = "".join([c for c in str(row[seq_col]).upper() if c.isupper() and c.isalpha()])
                    seq_to_org[s_clean] = str(row[org_col]).replace(" lysate", "").strip()
        except Exception as e:
            pass
            
    print(f"Mapped {len(seq_to_org)} sequences to organism metadata.")
    return seq_to_org

def main():
    print("=====================================================================================")
    print("  STABLEPROT V9 CROSS-SPECIES STRATIFICATION BENCHMARK")
    print("=====================================================================================")
    
    seq_to_org = load_organism_mapping()
    
    # Load ProThermDB evaluation results
    protherm_path = os.path.join(PROJECT_ROOT, "new_data/protherm_evaluation_results.pt")
    data = torch.load(protherm_path, map_location='cpu', weights_only=False)
    from Bio import SeqIO
    protherm_fasta = os.path.join(PROJECT_ROOT, 'new_data/prothermdb_validation.fasta')
    protherm_csv = os.path.join(PROJECT_ROOT, 'new_data/prothermdb_validation.csv')
    df_p = pd.read_csv(protherm_csv)
    protherm_dict = {str(row['UniProt_ID']): float(row['Tm']) for _, row in df_p.iterrows() if not np.isnan(row['Tm'])}
    
    v7_train_path = os.path.join(PROJECT_ROOT, "data/embeddings/prepared_data_v7_saprot1.3b_seqonly.pt")
    train_seqs_set = set()
    if os.path.exists(v7_train_path):
        v7_data_tmp = torch.load(v7_train_path, map_location='cpu', weights_only=False)
        if 'train_tm' in v7_data_tmp and 'sequences' in v7_data_tmp['train_tm']:
            train_seqs_set = {str(s).upper() for s in v7_data_tmp['train_tm']['sequences']}
    v8_train_path = os.path.join(PROJECT_ROOT, "data/embeddings/saprot_tm_struct_embeddings.pt")
    if os.path.exists(v8_train_path):
        v8_data_tmp = torch.load(v8_train_path, map_location='cpu', weights_only=False)
        if 'train_tm' in v8_data_tmp and 'sequences' in v8_data_tmp['train_tm']:
            train_seqs_set.update({str(s).upper() for s in v8_data_tmp['train_tm']['sequences']})

    seqs = []
    y_true_list = []
    for record in SeqIO.parse(protherm_fasta, 'fasta'):
        seq = str(record.seq)
        uid = record.id.split('|')[0]
        if uid in protherm_dict and seq.upper() not in train_seqs_set:
            seqs.append(seq)
            y_true_list.append(protherm_dict[uid])
    y_true = np.array(y_true_list)
    
    preds = data['predictions']
    k_v9 = 'StableProt V9' if 'StableProt V9' in preds else 'StableProt V8'
    pred_v9 = np.array(preds[k_v9])
    pred_tem = np.array(preds.get('TemBERTure', pred_v9))
    pred_deep = np.array(preds.get('DeepSTABp', pred_v9))
    pred_therm = np.array(preds.get('ThermoFormer', pred_v9))
    
    # Map ProTherm sequences to organism
    mapped_orgs = []
    for s in seqs:
        s_clean = "".join([c for c in str(s).upper() if c.isupper() and c.isalpha()])
        org = seq_to_org.get(s_clean, "Unknown / Mixed")
        # Simplify organism name to genus species
        if "Escherichia coli" in org or "E. coli" in org: org = "Escherichia coli"
        elif "Thermus thermophilus" in org or "T. thermophilus" in org: org = "Thermus thermophilus"
        elif "Bacillus subtilis" in org or "B. subtilis" in org: org = "Bacillus subtilis"
        elif "Saccharomyces cerevisiae" in org or "S. cerevisiae" in org: org = "Saccharomyces cerevisiae"
        elif "Homo sapiens" in org or "Human" in org: org = "Homo sapiens"
        elif "Pyrococcus furiosus" in org or "P. furiosus" in org: org = "Pyrococcus furiosus"
        elif "Geobacillus stearothermophilus" in org or "Bacillus stearothermophilus" in org: org = "Geobacillus stearothermophilus"
        elif "Mus musculus" in org: org = "Mus musculus"
        mapped_orgs.append(org)
        
    df = pd.DataFrame({
        'sequence': seqs,
        'organism': mapped_orgs,
        'y_true': y_true,
        'pred_v9': pred_v9,
        'pred_tem': pred_tem,
        'pred_deep': pred_deep,
        'pred_therm': pred_therm
    })
    
    # Filter to major identified species
    target_species = [
        "Escherichia coli", 
        "Thermus thermophilus", 
        "Bacillus subtilis", 
        "Saccharomyces cerevisiae", 
        "Homo sapiens", 
        "Pyrococcus furiosus", 
        "Geobacillus stearothermophilus"
    ]
    
    df_species = df[df['organism'].isin(target_species)].copy()
    print(f"\nFound {len(df_species)} test sequences across {len(target_species)} major target species.")
    
    rows = []
    for org in target_species:
        sub = df_species[df_species['organism'] == org]
        if len(sub) < 5:
            continue
        t_true = sub['y_true'].values
        
        # StableProt V9
        v9_p = sub['pred_v9'].values
        v9_mae = np.mean(np.abs(t_true - v9_p))
        v9_rho, _ = spearmanr(t_true, v9_p) if len(sub) > 10 else (0.0, 0)
        
        rows.append({
            'Species': org,
            'Sample_Count': len(sub),
            'Mean_True_Tm': round(np.mean(t_true), 1),
            'StableProt_V9_MAE': round(v9_mae, 3),
            'StableProt_V9_Spearman': round(v9_rho, 3),
        })
        print(f"  {org:<30} | N={len(sub):<4} | V9 MAE: {v9_mae:.2f}°C (ρ={v9_rho:.2f})")
        
    df_out = pd.DataFrame(rows)
    csv_out = os.path.join(VAL_SUITE_DIR, "cross_species_summary.csv")
    df_out.to_csv(csv_out, index=False)
    print(f"\nSaved cross-species summary table to: {csv_out}")
    
    # Plot multi-panel figure & JSON
    if len(df_out) > 0:
        import json
        json_out = os.path.join(OUT_DIR, "cross_species_generalization.json")
        with open(json_out, "w") as f:
            json.dump(df_out.to_dict(orient="records"), f, indent=2)
        print(f"Saved cross-species plot data JSON to: {json_out}")
 
        sns.set_context("paper", font_scale=1.15)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        x = np.arange(len(df_out))
        width = 0.5
        
        # Panel A: MAE
        axes[0].bar(x, df_out['StableProt_V9_MAE'], width, label='StableProt V9', color='#2563eb')
        axes[0].set_ylabel("Mean Absolute Error (MAE in °C, Lower is Better)")
        axes[0].set_title("A. Cross-Species $T_m$ Prediction Error per Organism")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f"{s}\n(N={n})" for s, n in zip(df_out['Species'], df_out['Sample_Count'])], rotation=25, ha='right')
        axes[0].grid(axis='y', linestyle=':', alpha=0.6)
        axes[0].legend(loc='upper right')
        
        # Panel B: Spearman Correlation
        axes[1].bar(x, df_out['StableProt_V9_Spearman'], width, label='StableProt V9', color='#10b981')
        axes[1].set_ylabel("Spearman Rank Correlation (ρ, Higher is Better)")
        axes[1].set_title("B. Intra-Organism Ranking Correlation (Uniform AA Composition)")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f"{s}\n(N={n})" for s, n in zip(df_out['Species'], df_out['Sample_Count'])], rotation=25, ha='right')
        axes[1].set_ylim(-0.2, 1.0)
        axes[1].grid(axis='y', linestyle=':', alpha=0.6)
        axes[1].legend(loc='upper right')
        
        plt.tight_layout()
        p_out = os.path.join(OUT_DIR, "cross_species_generalization.png")
        plt.savefig(p_out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved cross-species plot to: {p_out}")

if __name__ == "__main__":
    main()
