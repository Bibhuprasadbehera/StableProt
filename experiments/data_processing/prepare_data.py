"""
Data preparation script for TemStaPro experiments.

Parses FASTA files, generates ProtT5 embeddings, and saves processed
data as .pt files for training scripts to load.

Usage:
    python prepare_data.py --train-sample 5000 --val-sample 1000 --test-sample 2000
    python prepare_data.py --full    # Use all data (warning: very slow on CPU)
"""

import argparse
import os
import sys
import time
import torch

# Add experiments root to path
EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXPERIMENTS_DIR)

from common.data_utils import (
    parse_fasta_with_temps, sample_records, generate_embeddings,
    PROJECT_ROOT, DATASET_DIR, PROTTRANS_DIR
)


def main():
    parser = argparse.ArgumentParser(
        description='Prepare TemStaPro training data: parse FASTA + generate embeddings'
    )
    parser.add_argument('--train-sample', type=int, default=5000,
                        help='Number of training sequences to sample (default: 5000)')
    parser.add_argument('--val-sample', type=int, default=1000,
                        help='Number of validation sequences to sample (default: 1000)')
    parser.add_argument('--test-sample', type=int, default=2000,
                        help='Number of test sequences to sample (default: 2000)')
    parser.add_argument('--full', action='store_true',
                        help='Use all data (no sampling). WARNING: very slow on CPU!')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling (default: 42)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output .pt file path (default: experiments/prepared_data.pt)')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Directory to cache individual embeddings')
    parser.add_argument('--model-dir', type=str, default=None,
                        help='ProtTrans model directory')
    parser.add_argument('--max-seq-len', type=int, default=1500,
                        help='Maximum sequence length to include (default: 1500)')

    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(EXPERIMENTS_DIR, 'prepared_data.pt')
    if args.cache_dir is None:
        args.cache_dir = os.path.join(EXPERIMENTS_DIR, 'embeddings_cache')
    if args.model_dir is None:
        args.model_dir = PROTTRANS_DIR

    # ── FASTA file paths ──
    train_fasta = os.path.join(DATASET_DIR, 'TemStaPro-Major-30-imbal-training.fasta')
    val_fasta = os.path.join(DATASET_DIR, 'TemStaPro-Major-30-imbal-validation.fasta')
    test_fasta = os.path.join(DATASET_DIR, 'TemStaPro-Major-30-imbal-testing.fasta')

    for fpath in [train_fasta, val_fasta, test_fasta]:
        if not os.path.exists(fpath):
            print("ERROR: FASTA file not found: %s" % fpath, file=sys.stderr)
            sys.exit(1)

    # ── Parse FASTA files ──
    print("=" * 60)
    print("  TemStaPro Data Preparation")
    print("=" * 60)

    print("\n[1/4] Parsing FASTA files...")
    start = time.time()
    train_records = parse_fasta_with_temps(train_fasta)
    print("  Training: %d records parsed (%.1f sec)" % (len(train_records), time.time() - start))

    start = time.time()
    val_records = parse_fasta_with_temps(val_fasta)
    print("  Validation: %d records parsed (%.1f sec)" % (len(val_records), time.time() - start))

    start = time.time()
    test_records = parse_fasta_with_temps(test_fasta)
    print("  Testing: %d records parsed (%.1f sec)" % (len(test_records), time.time() - start))

    # ── Filter long sequences ──
    print("\n[2/4] Filtering sequences (max_seq_len=%d)..." % args.max_seq_len)
    train_records = [(s, seq, t) for s, seq, t in train_records if len(seq) <= args.max_seq_len]
    val_records = [(s, seq, t) for s, seq, t in val_records if len(seq) <= args.max_seq_len]
    test_records = [(s, seq, t) for s, seq, t in test_records if len(seq) <= args.max_seq_len]

    print("  After filtering: train=%d, val=%d, test=%d" % (
        len(train_records), len(val_records), len(test_records)))

    # ── Sample ──
    if not args.full:
        print("\n[3/4] Sampling (seed=%d)..." % args.seed)
        train_records = sample_records(train_records, args.train_sample, seed=args.seed)
        val_records = sample_records(val_records, args.val_sample, seed=args.seed + 1)
        test_records = sample_records(test_records, args.test_sample, seed=args.seed + 2)
        print("  Sampled: train=%d, val=%d, test=%d" % (
            len(train_records), len(val_records), len(test_records)))
    else:
        print("\n[3/4] Using ALL data (no sampling)")
        print("  WARNING: Embedding generation will be very slow on CPU!")

    # Print temperature stats for each split
    for name, records in [('Train', train_records), ('Val', val_records), ('Test', test_records)]:
        temps = [t for _, _, t in records]
        print("  %s temps: min=%.0f, max=%.0f, mean=%.1f, median=%.1f" % (
            name, min(temps), max(temps),
            sum(temps) / len(temps),
            sorted(temps)[len(temps) // 2]
        ))

    # ── Generate embeddings ──
    print("\n[4/4] Generating ProtT5 embeddings...")
    total_start = time.time()

    print("\n  --- Training embeddings ---")
    train_emb = generate_embeddings(train_records, model_dir=args.model_dir, cache_dir=args.cache_dir)

    print("\n  --- Validation embeddings ---")
    val_emb = generate_embeddings(val_records, model_dir=args.model_dir, cache_dir=args.cache_dir)

    print("\n  --- Test embeddings ---")
    test_emb = generate_embeddings(test_records, model_dir=args.model_dir, cache_dir=args.cache_dir)

    total_elapsed = time.time() - total_start
    print("\n  Total embedding time: %.1f seconds" % total_elapsed)

    # ── Save ──
    train_temps = [t for _, _, t in train_records]
    val_temps = [t for _, _, t in val_records]
    test_temps = [t for _, _, t in test_records]

    train_ids = [s for s, _, _ in train_records]
    val_ids = [s for s, _, _ in val_records]
    test_ids = [s for s, _, _ in test_records]

    save_data = {
        'train_embeddings': train_emb,
        'train_temps': train_temps,
        'train_ids': train_ids,
        'val_embeddings': val_emb,
        'val_temps': val_temps,
        'val_ids': val_ids,
        'test_embeddings': test_emb,
        'test_temps': test_temps,
        'test_ids': test_ids,
        'metadata': {
            'train_sample': len(train_records),
            'val_sample': len(val_records),
            'test_sample': len(test_records),
            'max_seq_len': args.max_seq_len,
            'seed': args.seed,
            'embedding_dim': 1024,
        }
    }

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(save_data, args.output)
    file_size = os.path.getsize(args.output) / (1024 * 1024)
    print("\n  Data saved to: %s (%.1f MB)" % (args.output, file_size))
    print("\n" + "=" * 60)
    print("  DONE! You can now run training with:")
    print("    cd experiments/v1_baseline && python train.py")
    print("    cd experiments/v2_improved && python train.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
