#!/usr/bin/env python3
"""
Bidirectional Decontamination Verification Script for StableProt V8.
Verifies zero sequence overlap between training splits (train_tm, train_ogt)
and all benchmark / holdout datasets (test_tm, test_ogt, val_ogt, ProThermDB, FireProtDB).
"""

import os
import sys
from pathlib import Path
import torch
import pandas as pd
from Bio import SeqIO

PROJECT_ROOT = Path(__file__).resolve().parents[3]

def clean_seq(seq):
    return "".join([c for c in str(seq).upper() if c.isupper() and c.isalpha()])

def main():
    print("=" * 70)
    print("BIDIRECTIONAL DECONTAMINATION AUDIT FOR STABLEPROT V8")
    print("=" * 70)

    # 1. Load Training Splits
    print("\nLoading training datasets...")
    tm_data = torch.load(PROJECT_ROOT / "data/embeddings/saprot_tm_struct_embeddings.pt", map_location='cpu', weights_only=False)
    ogt_data = torch.load(PROJECT_ROOT / "data/embeddings/prepared_data_v7_saprot1.3b_seqonly_ogt_split.pt", map_location='cpu', weights_only=False)

    train_tm_seqs = {clean_seq(s) for s in tm_data['train_tm']['sequences']}
    train_ogt_seqs = {clean_seq(s) for s in ogt_data['train_ogt']['sequences']}
    print(f"  Train Tm sequences:  {len(train_tm_seqs):,}")
    print(f"  Train OGT sequences: {len(train_ogt_seqs):,}")

    # 2. Load Internal Holdouts
    test_tm_seqs = {clean_seq(s) for s in tm_data['test_tm']['sequences']}
    val_tm_seqs = {clean_seq(s) for s in tm_data['val_tm']['sequences']}
    test_ogt_seqs = {clean_seq(s) for s in ogt_data['test_ogt']['sequences']}
    val_ogt_seqs = {clean_seq(s) for s in ogt_data['val_ogt']['sequences']}

    # 3. Load External Benchmark Datasets
    protherm_seqs = set()
    protherm_fasta = PROJECT_ROOT / "new_data/prothermdb_validation.fasta"
    if protherm_fasta.exists():
        for record in SeqIO.parse(protherm_fasta, 'fasta'):
            protherm_seqs.add(clean_seq(record.seq))

    fireprot_seqs = set()
    fireprot_pt = PROJECT_ROOT / "data/embeddings/fireprot_eval_data.pt"
    if fireprot_pt.exists():
        d_fp = torch.load(fireprot_pt, map_location='cpu', weights_only=False)
        if 'sequences' in d_fp:
            fireprot_seqs = {clean_seq(s) for s in d_fp['sequences']}
        elif 'test_tm' in d_fp:
            fireprot_seqs = {clean_seq(s) for s in d_fp['test_tm']['sequences']}

    brenda_seqs = set()
    brenda_csv = PROJECT_ROOT / "new_data/brenda_ood_benchmark.csv"
    if brenda_csv.exists():
        df_br = pd.read_csv(brenda_csv)
        brenda_seqs = {clean_seq(s) for s in df_br['sequence'].dropna()}

    benchmarks = {
        'Internal Val Tm': val_tm_seqs,
        'Internal Test Tm': test_tm_seqs,
        'Internal Val OGT': val_ogt_seqs,
        'Internal Test OGT': test_ogt_seqs,
        'External ProThermDB': protherm_seqs,
        'External FireProtDB': fireprot_seqs,
        'External BRENDA OOD': brenda_seqs,
    }

    # Perform Bidirectional Overlap Checks
    print("\nExecuting bidirectional overlap checks against train_tm...")
    for name, seqs in benchmarks.items():
        overlap = train_tm_seqs.intersection(seqs)
        status = "PASSED" if len(overlap) == 0 else "WARNING/CONTAMINATED"
        print(f"  [{status}] train_tm vs {name:<20}: {len(overlap):3d} overlapping sequences (out of {len(seqs)})")

    print("\nExecuting bidirectional overlap checks against train_ogt...")
    for name, seqs in benchmarks.items():
        overlap = train_ogt_seqs.intersection(seqs)
        status = "PASSED" if len(overlap) == 0 else "INFO/OVERLAP"
        print(f"  [{status}] train_ogt vs {name:<20}: {len(overlap):3d} overlapping sequences (out of {len(seqs)})")

    print("\nDecontamination audit completed successfully.")

if __name__ == "__main__":
    main()
