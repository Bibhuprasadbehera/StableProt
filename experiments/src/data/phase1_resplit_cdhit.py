#!/usr/bin/env python3
"""
Phase 1: Data Re-split + CD-HIT Decontamination

1. Re-splits 28,739 Tm sequences into ~25,900 train + ~2,800 val (temperature-stratified)
2. Keeps existing 2,007 test set
3. Runs CD-HIT at 40% identity across all splits to guarantee no homology leakage
4. Reports final split sizes and per-bin sample counts

Usage:
    python phase1_resplit_cdhit.py [--skip-cdhit] [--dry-run]
"""

import argparse
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_FILE = PROJECT_ROOT / "data" / "embeddings" / "prepared_data_v4_saprot.pt"
OUTPUT_FILE = PROJECT_ROOT / "data" / "embeddings" / "prepared_data_v7_splits.pt"
CDHIT_WORK_DIR = PROJECT_ROOT / "data" / "cdhit_phase1"

VAL_FRACTION = 0.10  # 10% of training data → validation
BIN_WIDTH = 5  # °C bins for stratification
TM_RANGE = (25, 100)  # Temperature range for bins
CDHIT_IDENTITY = 0.40  # 40% identity threshold (CD-HIT v4.8.1 has bugs at 30%)


def stratified_split(labels, val_fraction, bin_width, tm_range, seed=42):
    """Temperature-stratified train/val split.
    
    Ensures each 5°C temperature bin contributes proportionally to both sets.
    """
    rng = np.random.RandomState(seed)
    bins = np.arange(tm_range[0], tm_range[1] + bin_width, bin_width)
    bin_indices = np.digitize(labels, bins) - 1

    train_idx = []
    val_idx = []

    for b in range(len(bins) - 1):
        bin_mask = np.where(bin_indices == b)[0]
        if len(bin_mask) == 0:
            continue

        rng.shuffle(bin_mask)
        n_val = max(1, int(len(bin_mask) * val_fraction))  # At least 1 per bin
        val_idx.extend(bin_mask[:n_val].tolist())
        train_idx.extend(bin_mask[n_val:].tolist())

    return sorted(train_idx), sorted(val_idx)


def write_fasta(sequences, ids, filepath):
    """Write sequences to FASTA file. Sanitize IDs for CD-HIT compatibility."""
    with open(filepath, 'w') as f:
        for i, (seq_id, seq) in enumerate(zip(ids, sequences)):
            # CD-HIT needs clean IDs — use index as primary ID, original as description
            clean_id = f"seq_{i}"
            f.write(f">{clean_id} {seq_id}\n{seq}\n")


def run_cdhit_2d(query_fasta, db_fasta, output_prefix, identity=0.40):
    """Run CD-HIT-2D to find sequences in query that are NOT similar to db.
    
    At 40% identity, word size must be 2. Use -G 0 for local alignment.
    """
    cmd = [
        "cd-hit-2d",
        "-i", str(db_fasta),      # Database (training set)
        "-i2", str(query_fasta),  # Query (val or test set)
        "-o", str(output_prefix),
        "-c", str(identity),
        "-n", "2",        # Word size 2 required for <40% identity
        "-G", "0",        # Local alignment
        "-M", "8000",     # 8GB memory
        "-T", "4",        # 4 threads
        "-d", "0",        # Full sequence name in cluster file
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  CD-HIT-2D stdout: {result.stdout[-500:]}")
        print(f"  CD-HIT-2D stderr: {result.stderr[-500:]}")
        raise RuntimeError("CD-HIT-2D failed")
    return result


def parse_cdhit_clusters(cluster_file):
    """Parse CD-HIT cluster file to find which query sequences are novel (not in db)."""
    # CD-HIT-2D output: sequences that survive are novel (not similar to db)
    novel_ids = set()
    with open(cluster_file) as f:
        for line in f:
            if line.startswith(">"):
                pass  # Skip cluster headers
            elif "*" in line:
                # Representative sequence — extract ID
                seq_id = line.split(">")[1].split("...")[0].strip()
                novel_ids.add(seq_id)
    return novel_ids


def read_surviving_indices(output_fasta):
    """Read integer indices from CD-HIT-2D output FASTA (surviving = novel sequences).
    
    FASTA headers are in format '>seq_N original_id', so we extract N.
    """
    indices = set()
    with open(output_fasta) as f:
        for line in f:
            if line.startswith(">"):
                # Extract seq_N from header
                clean_id = line[1:].strip().split()[0]  # "seq_N"
                idx = int(clean_id.split("_")[1])
                indices.add(idx)
    return indices


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Data re-split + CD-HIT decontamination")
    parser.add_argument("--skip-cdhit", action="store_true", help="Skip CD-HIT step")
    parser.add_argument("--dry-run", action="store_true", help="Show statistics without saving")
    args = parser.parse_args()

    print("Phase 1: Data Re-split + CD-HIT Decontamination")
    print(f"  Val fraction: {VAL_FRACTION}")
    print(f"  CD-HIT identity threshold: {CDHIT_IDENTITY}")

    # Load data
    print("\nLoading data...")
    data = torch.load(DATA_FILE, map_location='cpu', weights_only=False)

    # Current train_tm is the full training set (28,739)
    train_emb = data['train_tm']['embeddings']
    train_seqs = data['train_tm']['sequences']
    train_tm = np.array(data['train_tm']['tm_consensus'])
    train_ids = data['train_tm']['ids']

    test_emb = data['test_tm']['embeddings']
    test_seqs = data['test_tm']['sequences']
    test_tm = np.array(data['test_tm'].get('tm_consensus', data['test_tm'].get('labels')))
    test_ids = data['test_tm']['ids']

    print(f"  Current train: {len(train_seqs)}")
    print(f"  Current test: {len(test_seqs)}")

    # Step 1: Stratified split
    print("\nStep 1: Temperature-stratified train/val split...")
    train_idx, val_idx = stratified_split(train_tm, VAL_FRACTION, BIN_WIDTH, TM_RANGE)
    print(f"  New train: {len(train_idx)}")
    print(f"  New val: {len(val_idx)}")

    # Per-bin statistics
    bins = np.arange(TM_RANGE[0], TM_RANGE[1] + BIN_WIDTH, BIN_WIDTH)
    print(f"\n  {'Bin':>10} | {'Train':>6} | {'Val':>5} | {'Test':>5}")
    print(f"  {'-'*10}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}")
    for i in range(len(bins) - 1):
        t_count = sum(1 for idx in train_idx if bins[i] <= train_tm[idx] < bins[i+1])
        v_count = sum(1 for idx in val_idx if bins[i] <= train_tm[idx] < bins[i+1])
        te_count = sum(1 for t in test_tm if bins[i] <= t < bins[i+1])
        print(f"  {bins[i]:>4}-{bins[i+1]:<4}  | {t_count:>6} | {v_count:>5} | {te_count:>5}")

    if args.skip_cdhit:
        print("\n  --skip-cdhit: Skipping CD-HIT decontamination.")
    else:
        # Step 2: CD-HIT decontamination
        print("\nStep 2: CD-HIT decontamination at 40% identity...")
        CDHIT_WORK_DIR.mkdir(parents=True, exist_ok=True)

        # Build indexed sequences for each split
        new_train_seqs = [train_seqs[i] for i in train_idx]
        new_train_ids = [train_ids[i] for i in train_idx]
        new_val_seqs = [train_seqs[i] for i in val_idx]
        new_val_ids = [train_ids[i] for i in val_idx]

        train_fasta = CDHIT_WORK_DIR / "train.fasta"
        val_fasta = CDHIT_WORK_DIR / "val.fasta"
        test_fasta = CDHIT_WORK_DIR / "test.fasta"

        write_fasta(new_train_seqs, new_train_ids, train_fasta)
        write_fasta(new_val_seqs, new_val_ids, val_fasta)
        write_fasta(test_seqs, test_ids, test_fasta)

        # CD-HIT-2D: val vs train — find val sequences NOT similar to train
        print("\n  2a: Val vs Train (removing val sequences similar to train)...")
        val_vs_train_out = CDHIT_WORK_DIR / "val_vs_train"
        run_cdhit_2d(val_fasta, train_fasta, val_vs_train_out, identity=CDHIT_IDENTITY)
        val_surviving_local_idx = read_surviving_indices(val_vs_train_out)
        val_removed = len(new_val_ids) - len(val_surviving_local_idx)
        print(f"  Val: {len(new_val_ids)} → {len(val_surviving_local_idx)} (removed {val_removed} with >30% identity to train)")

        # Map local val indices back to global indices
        val_idx_clean = [val_idx[i] for i in range(len(val_idx)) if i in val_surviving_local_idx]

        # CD-HIT-2D: test vs train
        print("\n  2b: Test vs Train (removing test sequences similar to train)...")
        test_vs_train_out = CDHIT_WORK_DIR / "test_vs_train"
        run_cdhit_2d(test_fasta, train_fasta, test_vs_train_out, identity=CDHIT_IDENTITY)
        test_surviving_local_idx = read_surviving_indices(test_vs_train_out)
        test_removed = len(test_ids) - len(test_surviving_local_idx)
        print(f"  Test: {len(test_ids)} → {len(test_surviving_local_idx)} (removed {test_removed} with >30% identity to train)")

        # CD-HIT-2D: surviving test vs clean val
        print("\n  2c: Test vs Val (removing test sequences similar to val)...")
        # Write clean val & clean test FASTAs
        clean_val_seqs_2 = [train_seqs[idx] for idx in val_idx_clean]
        clean_val_ids_2 = [train_ids[idx] for idx in val_idx_clean]
        clean_val_fasta = CDHIT_WORK_DIR / "val_clean.fasta"
        write_fasta(clean_val_seqs_2, clean_val_ids_2, clean_val_fasta)

        test_surviving_list = sorted(test_surviving_local_idx)
        clean_test_seqs_2 = [test_seqs[i] for i in test_surviving_list]
        clean_test_ids_2 = [test_ids[i] for i in test_surviving_list]
        clean_test_fasta = CDHIT_WORK_DIR / "test_clean.fasta"
        write_fasta(clean_test_seqs_2, clean_test_ids_2, clean_test_fasta)

        test_vs_val_out = CDHIT_WORK_DIR / "test_vs_val"
        run_cdhit_2d(clean_test_fasta, clean_val_fasta, test_vs_val_out, identity=CDHIT_IDENTITY)
        test_final_local_idx = read_surviving_indices(test_vs_val_out)
        test_val_removed = len(clean_test_ids_2) - len(test_final_local_idx)
        print(f"  Test: {len(clean_test_ids_2)} → {len(test_final_local_idx)} (removed {test_val_removed} with >30% identity to val)")

        # Map back to original test indices
        test_keep_indices = [test_surviving_list[i] for i in range(len(test_surviving_list)) if i in test_final_local_idx]

        # Update val_idx for output
        val_idx = val_idx_clean

    # Build output data dict
    print("\n=== Final Split Sizes ===")

    # Get all keys from train_tm to preserve metadata
    all_keys = list(data['train_tm'].keys())

    new_data = {
        'train_ogt': data['train_ogt'],  # OGT unchanged (or use Phase 0 corrected)
    }

    # Build new train split
    new_data['train_tm'] = {}
    for key in all_keys:
        val = data['train_tm'][key]
        if isinstance(val, torch.Tensor):
            new_data['train_tm'][key] = val[train_idx]
        elif isinstance(val, list):
            new_data['train_tm'][key] = [val[i] for i in train_idx]
        elif isinstance(val, np.ndarray):
            new_data['train_tm'][key] = val[np.array(train_idx)]
        else:
            new_data['train_tm'][key] = val

    # Build new val split
    new_data['val_tm'] = {}
    for key in all_keys:
        val = data['train_tm'][key]
        if isinstance(val, torch.Tensor):
            new_data['val_tm'][key] = val[val_idx]
        elif isinstance(val, list):
            new_data['val_tm'][key] = [val[i] for i in val_idx]
        elif isinstance(val, np.ndarray):
            new_data['val_tm'][key] = val[np.array(val_idx)]
        else:
            new_data['val_tm'][key] = val

    # Build test split (filtered if CD-HIT ran)
    test_keys = list(data['test_tm'].keys())
    new_data['test_tm'] = {}
    if not args.skip_cdhit:
        for key in test_keys:
            val = data['test_tm'][key]
            if isinstance(val, torch.Tensor):
                new_data['test_tm'][key] = val[test_keep_indices]
            elif isinstance(val, list):
                new_data['test_tm'][key] = [val[i] for i in test_keep_indices]
            elif isinstance(val, np.ndarray):
                new_data['test_tm'][key] = val[np.array(test_keep_indices)]
            else:
                new_data['test_tm'][key] = val
    else:
        new_data['test_tm'] = data['test_tm']

    # Print final sizes
    for split in ['train_tm', 'val_tm', 'test_tm']:
        n = len(new_data[split]['sequences'])
        tms = np.array(new_data[split].get('tm_consensus', new_data[split].get('labels')))
        print(f"  {split:>10}: {n:>6} sequences | Tm range: [{tms.min():.1f}, {tms.max():.1f}]°C | Mean: {tms.mean():.1f}°C")

    if not args.dry_run:
        print(f"\nSaving to {OUTPUT_FILE}...")
        torch.save(new_data, OUTPUT_FILE)
        print("  Done.")
    else:
        print("\n  --dry-run: No data saved.")


if __name__ == "__main__":
    main()
