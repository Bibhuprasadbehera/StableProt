#!/usr/bin/env python3
"""
Step 7: FLIP Meltome Benchmark — Download, Decontaminate, Prepare

1. Downloads FLIP meltome_mixed_split from HuggingFace
2. CD-HIT at 40% identity against V7 training sequences
3. Outputs clean test set for evaluation

Usage:
    python step7_flip_meltome.py [--skip-download] [--skip-cdhit]
"""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FLIP_DIR = PROJECT_ROOT / "data" / "flip_meltome"
CDHIT_WORK = FLIP_DIR / "cdhit_decontam"
V7_DATA = PROJECT_ROOT / "data" / "embeddings" / "prepared_data_v7_saprot1.3b_seqonly.pt"
OUTPUT = FLIP_DIR / "flip_clean.csv"


def download_flip():
    """Download FLIP meltome mixed_split from HuggingFace."""
    FLIP_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = FLIP_DIR / "mixed_split.csv"

    if csv_path.exists():
        print(f"  Already downloaded: {csv_path}")
        return csv_path

    print("  Downloading FLIP meltome mixed_split from HuggingFace...")
    url = "https://huggingface.co/datasets/hazemessam/meltome/resolve/main/mixed_split.csv"
    try:
        df = pd.read_csv(url)
        df.to_csv(csv_path, index=False)
        print(f"  Saved {len(df)} rows to {csv_path}")
    except Exception as e:
        # Fallback: try FLIP github
        print(f"  HF failed ({e}), trying FLIP GitHub...")
        url2 = "https://raw.githubusercontent.com/J-SNACKKB/FLIP/main/splits/meltome/splits/mixed_split.csv"
        df = pd.read_csv(url2)
        df.to_csv(csv_path, index=False)
        print(f"  Saved {len(df)} rows to {csv_path}")

    return csv_path


def extract_test_set(csv_path):
    """Extract test-set sequences and labels from FLIP split."""
    df = pd.read_csv(csv_path)
    print(f"\n  FLIP columns: {list(df.columns)}")
    print(f"  Total rows: {len(df)}")

    # FLIP uses 'set' column: train/test
    if 'set' in df.columns:
        test_df = df[df['set'] == 'test'].copy()
    elif 'split' in df.columns:
        test_df = df[df['split'] == 'test'].copy()
    else:
        print(f"  WARNING: No 'set' or 'split' column. Available: {list(df.columns)}")
        return None

    # FLIP meltome uses 'sequence' and 'label' columns
    seq_col = 'sequence' if 'sequence' in test_df.columns else 'seq'
    target_col = 'label' if 'label' in test_df.columns else 'target'

    test_df = test_df.dropna(subset=[seq_col, target_col])
    print(f"  Test set: {len(test_df)} sequences")
    print(f"  Target range: [{test_df[target_col].min():.1f}, {test_df[target_col].max():.1f}]")

    return test_df, seq_col, target_col


def write_fasta(sequences, ids, filepath):
    """Write sequences to FASTA."""
    with open(filepath, 'w') as f:
        for i, (sid, seq) in enumerate(zip(ids, sequences)):
            f.write(f">seq_{i} {sid}\n{seq}\n")


def decontaminate(test_df, seq_col, target_col):
    """CD-HIT at 40% identity against V7 training sequences."""
    CDHIT_WORK.mkdir(parents=True, exist_ok=True)

    # Load training sequences
    print("\n  Loading V7 training sequences...")
    data = torch.load(V7_DATA, map_location='cpu', weights_only=False)
    train_seqs = list(data['train_tm']['sequences'])
    train_ids = [f"train_{i}" for i in range(len(train_seqs))]
    print(f"  Training sequences: {len(train_seqs)}")

    # Write FASTAs
    train_fasta = CDHIT_WORK / "train.fasta"
    test_fasta = CDHIT_WORK / "flip_test.fasta"

    test_seqs = test_df[seq_col].tolist()
    test_ids = [f"flip_{i}" for i in range(len(test_seqs))]

    write_fasta(train_seqs, train_ids, train_fasta)
    write_fasta(test_seqs, test_ids, test_fasta)

    # CD-HIT-2D: find FLIP test sequences NOT similar to training
    output_prefix = CDHIT_WORK / "flip_vs_train"
    cmd = [
        "cd-hit-2d",
        "-i", str(train_fasta),     # Database
        "-i2", str(test_fasta),     # Query
        "-o", str(output_prefix),
        "-c", "0.40",               # 40% identity
        "-n", "2",                  # Word size for <40%
        "-M", "8000",
        "-T", "4",
        "-d", "0",
    ]
    print(f"\n  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-500:]}")
        raise RuntimeError("CD-HIT-2D failed")

    # Read surviving sequences
    surviving_indices = set()
    with open(output_prefix) as f:
        for line in f:
            if line.startswith(">"):
                clean_id = line[1:].strip().split()[0]
                idx = int(clean_id.split("_")[1])
                surviving_indices.add(idx)

    removed = len(test_seqs) - len(surviving_indices)
    print(f"\n  FLIP test: {len(test_seqs)} → {len(surviving_indices)} "
          f"(removed {removed} with >40% identity to training)")

    # Filter dataframe
    clean_mask = [i in surviving_indices for i in range(len(test_df))]
    clean_df = test_df.iloc[clean_mask].copy()

    del data
    return clean_df


def main():
    parser = argparse.ArgumentParser(description="FLIP Meltome: Download + Decontaminate")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-cdhit", action="store_true")
    args = parser.parse_args()

    print("Step 7: FLIP Meltome Benchmark Preparation")
    print("=" * 60)

    # Download
    if not args.skip_download:
        csv_path = download_flip()
    else:
        csv_path = FLIP_DIR / "mixed_split.csv"

    # Extract test set
    result = extract_test_set(csv_path)
    if result is None:
        return
    test_df, seq_col, target_col = result

    # Decontaminate
    if not args.skip_cdhit:
        clean_df = decontaminate(test_df, seq_col, target_col)
    else:
        clean_df = test_df
        print("  --skip-cdhit: skipping decontamination")

    # Save clean test set
    clean_df.to_csv(OUTPUT, index=False)
    print(f"\n  Saved clean FLIP test set: {OUTPUT}")
    print(f"  Clean sequences: {len(clean_df)}")
    print(f"  Original test:   {len(test_df)}")
    print(f"  Removed:         {len(test_df) - len(clean_df)} ({100*(len(test_df)-len(clean_df))/len(test_df):.1f}%)")


if __name__ == "__main__":
    main()
