"""
Shared data utilities for TemStaPro experiments.

Handles FASTA parsing, temperature extraction, embedding generation/caching,
and PyTorch dataset creation.
"""

import os
import sys
import random
import time
import torch
import numpy as np
from hashlib import sha256
from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler

# ── Paths ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STABLEPROT_DIR = os.path.join(PROJECT_ROOT, "StableProt")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
PROTTRANS_DIR = os.path.join(STABLEPROT_DIR, "ProtTrans")

# ──────────────────────────────────────────
# FASTA Parsing
# ──────────────────────────────────────────

def parse_fasta_with_temps(fasta_path):
    """
    Parse FASTA file and extract sequences with OGT from headers.
    Header format: >taxid|uniprot_id|temperature

    Returns:
        list of tuples: (seq_id, sequence, temperature)
    """
    records = []
    current_id = None
    current_temp = None
    current_seq_parts = []

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save previous record
                if current_id is not None:
                    seq = ''.join(current_seq_parts)
                    records.append((current_id, seq, current_temp))

                parts = line[1:].split('|')
                # Use taxid|uniprot as the unique ID
                if len(parts) >= 2:
                    current_id = "%s|%s" % (parts[0], parts[1])
                else:
                    current_id = parts[0]

                try:
                    current_temp = float(parts[2]) if len(parts) >= 3 else None
                except ValueError:
                    current_temp = None

                current_seq_parts = []
            else:
                current_seq_parts.append(line)

    # Don't forget the last record
    if current_id is not None:
        seq = ''.join(current_seq_parts)
        records.append((current_id, seq, current_temp))

    # Filter out records without temperature
    records = [(sid, seq, temp) for sid, seq, temp in records if temp is not None]
    return records


def sample_records(records, n, seed=42):
    """Randomly sample n records from list. Returns all if n >= len(records)."""
    random.seed(seed)
    if n >= len(records):
        return list(records)
    return random.sample(records, n)


def create_binary_labels(temperatures, threshold):
    """Convert list of temperatures to binary tensor: 1 if temp >= threshold."""
    return torch.tensor(
        [1.0 if t >= threshold else 0.0 for t in temperatures],
        dtype=torch.float32
    )


# ──────────────────────────────────────────
# Embedding Generation
# ──────────────────────────────────────────

def _preprocess_sequence(seq):
    """Clean sequence for ProtT5: replace non-standard amino acids."""
    seq = seq.upper().replace('-', '')
    seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
    return seq


def generate_embeddings(records, model_dir=None, cache_dir=None, device=None,
                        chunk_size=10000):
    """
    Generate ProtT5 mean embeddings for a list of (seq_id, sequence, temp) records.

    Processes in chunks to handle large datasets (900K+ sequences).
    Caches individual embeddings so the process is resumable.

    Args:
        records: list of (seq_id, sequence, temperature) tuples
        model_dir: path to ProtTrans model directory (default: StableProt/ProtTrans)
        cache_dir: directory to cache individual embeddings (optional)
        device: 'cpu', 'cuda', or None (auto-detect)
        chunk_size: number of sequences to process per chunk (default: 10000)

    Returns:
        torch.Tensor of shape (n_seqs, 1024) — mean embeddings
    """
    if model_dir is None:
        model_dir = PROTTRANS_DIR
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    # Check which sequences need fresh embeddings
    embeddings_dict = {}
    needs_generation = []

    print("  Checking cache for %d sequences..." % len(records))
    for seq_id, seq, temp in records:
        clean_seq = _preprocess_sequence(seq)
        seq_hash = sha256(clean_seq.encode('utf-8')).hexdigest()
        cache_path = None

        if cache_dir:
            cache_path = os.path.join(cache_dir, "mean_%s.pt" % seq_hash)
            if os.path.exists(cache_path):
                cached = torch.load(cache_path, map_location='cpu')
                embeddings_dict[seq_id] = cached['mean_representations'].flatten()
                continue

        needs_generation.append((seq_id, clean_seq, temp, cache_path))

    print("  Cached: %d, Need generation: %d" % (len(embeddings_dict), len(needs_generation)))

    if needs_generation:
        # Import ProtTrans
        sys.path.insert(0, STABLEPROT_DIR)
        from prottrans_models import load_model_and_tokenizer, get_embeddings

        print("  Loading ProtT5 model from %s ..." % model_dir)
        print("  Device: %s" % device)
        model, tokenizer = load_model_and_tokenizer(
            model_dir,
            "Rostlab/prot_t5_xl_half_uniref50-enc"
        )
        # Move model to correct device (load_model_and_tokenizer auto-detects)
        print("  Model loaded on: %s" % next(model.parameters()).device)

        total_seqs = len(needs_generation)
        total_generated = 0
        overall_start = time.time()

        # Process in chunks
        n_chunks = (total_seqs + chunk_size - 1) // chunk_size
        print("  Processing %d sequences in %d chunks of %d..." % (
            total_seqs, n_chunks, chunk_size))

        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, total_seqs)
            chunk = needs_generation[chunk_start:chunk_end]

            # Build sequences dict for this chunk
            seqs_dict = {}
            for seq_id, clean_seq, temp, _ in chunk:
                seqs_dict[seq_id] = clean_seq

            chunk_time_start = time.time()
            results = get_embeddings(
                model, tokenizer, seqs_dict,
                per_residue=False, per_protein=True
            )
            chunk_elapsed = time.time() - chunk_time_start
            total_generated += len(results['mean_representations'])

            # Cache results
            for seq_id, clean_seq, temp, cache_path in chunk:
                if seq_id in results['mean_representations']:
                    emb = torch.from_numpy(results['mean_representations'][seq_id]).flatten()
                    embeddings_dict[seq_id] = emb

                    if cache_path:
                        torch.save({
                            'label': seq_id,
                            'sequence': clean_seq,
                            'mean_representations': emb,
                        }, cache_path)

            # Progress report
            elapsed_total = time.time() - overall_start
            rate = total_generated / max(elapsed_total, 1)
            remaining = total_seqs - total_generated
            eta_seconds = remaining / max(rate, 0.01)
            eta_min = eta_seconds / 60

            print("  [Chunk %d/%d] %d/%d done (%.1f seq/s) | "
                  "Chunk: %.0fs | Elapsed: %.0fs | ETA: %.0f min" % (
                      chunk_idx + 1, n_chunks,
                      total_generated, total_seqs,
                      rate, chunk_elapsed, elapsed_total, eta_min))

        # Free GPU memory
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        total_time = time.time() - overall_start
        print("  DONE! Generated %d embeddings in %.1f seconds (%.1f min)" % (
            total_generated, total_time, total_time / 60))

    # Assemble in original order
    embedding_list = []
    missing_count = 0
    for seq_id, seq, temp in records:
        if seq_id in embeddings_dict:
            embedding_list.append(embeddings_dict[seq_id])
        else:
            missing_count += 1
            embedding_list.append(torch.zeros(1024))

    if missing_count > 0:
        print("  WARNING: %d sequences missing embeddings (using zeros)" % missing_count)

    return torch.stack(embedding_list)


# ──────────────────────────────────────────
# Dataset & DataLoader utilities
# ──────────────────────────────────────────

class TemStaProDataset(Dataset):
    """Simple dataset wrapping embeddings and binary labels."""

    def __init__(self, embeddings, labels):
        assert len(embeddings) == len(labels), \
            "Embeddings (%d) and labels (%d) size mismatch" % (len(embeddings), len(labels))
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def get_balanced_sampler(labels):
    """
    Create a WeightedRandomSampler that balances classes during training.
    Each batch will have approximately equal positive/negative samples.
    """
    labels_long = labels.long()
    class_counts = torch.bincount(labels_long)

    # Avoid division by zero
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[labels_long]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def create_data_loaders(train_emb, train_labels, val_emb, val_labels,
                        batch_size=64, balanced=False):
    """
    Create training and validation DataLoaders.

    Args:
        train_emb: training embeddings tensor
        train_labels: training labels tensor
        val_emb: validation embeddings tensor
        val_labels: validation labels tensor
        batch_size: batch size
        balanced: whether to use balanced sampling for training

    Returns:
        (train_loader, val_loader)
    """
    train_dataset = TemStaProDataset(train_emb, train_labels)
    val_dataset = TemStaProDataset(val_emb, val_labels)

    if balanced:
        sampler = get_balanced_sampler(train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  sampler=sampler, drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, drop_last=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# ──────────────────────────────────────────
# High-level data preparation
# ──────────────────────────────────────────

def prepare_data_for_threshold(prepared_data_path, threshold):
    """
    Load preprocessed data and create binary labels for a specific threshold.

    Args:
        prepared_data_path: path to .pt file from prepare_data.py
        threshold: temperature threshold for binary classification

    Returns:
        dict with keys: train_emb, train_labels, val_emb, val_labels,
                        test_emb, test_labels, train_temps, val_temps, test_temps
    """
    data = torch.load(prepared_data_path)

    result = {}
    for split in ['train', 'val', 'test']:
        result['%s_emb' % split] = data['%s_embeddings' % split]
        temps = data['%s_temps' % split]
        result['%s_temps' % split] = temps
        result['%s_labels' % split] = create_binary_labels(temps, threshold)

        n_pos = int(result['%s_labels' % split].sum().item())
        n_total = len(result['%s_labels' % split])
        print("  %s split — threshold %d°C: %d/%d positive (%.1f%%)" % (
            split.capitalize(), threshold, n_pos, n_total, 100.0 * n_pos / n_total
        ))

    return result
