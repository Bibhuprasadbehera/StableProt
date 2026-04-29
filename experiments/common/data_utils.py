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


def generate_embeddings(records, model_dir=None, cache_dir=None, device='cpu'):
    """
    Generate ProtT5 mean embeddings for a list of (seq_id, sequence, temp) records.

    Args:
        records: list of (seq_id, sequence, temperature) tuples
        model_dir: path to ProtTrans model directory (default: StableProt/ProtTrans)
        cache_dir: directory to cache individual embeddings (optional)
        device: 'cpu' or 'cuda'

    Returns:
        torch.Tensor of shape (n_seqs, 1024) — mean embeddings
    """
    if model_dir is None:
        model_dir = PROTTRANS_DIR

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    # Check which sequences need fresh embeddings
    embeddings_dict = {}
    needs_generation = []

    for seq_id, seq, temp in records:
        clean_seq = _preprocess_sequence(seq)
        seq_hash = sha256(clean_seq.encode('utf-8')).hexdigest()
        cache_path = None

        if cache_dir:
            cache_path = os.path.join(cache_dir, "mean_%s.pt" % seq_hash)
            if os.path.exists(cache_path):
                cached = torch.load(cache_path)
                embeddings_dict[seq_id] = cached['mean_representations'].flatten()
                continue

        needs_generation.append((seq_id, clean_seq, temp, cache_path))

    print("  Cached: %d, Need generation: %d" % (len(embeddings_dict), len(needs_generation)))

    if needs_generation:
        # Import ProtTrans
        sys.path.insert(0, STABLEPROT_DIR)
        from prottrans_models import load_model_and_tokenizer, get_embeddings, save_embeddings

        print("  Loading ProtT5 model from %s ..." % model_dir)
        model, tokenizer = load_model_and_tokenizer(
            model_dir,
            "Rostlab/prot_t5_xl_half_uniref50-enc"
        )
        print("  Model loaded. Generating embeddings...")

        # Build sequences dict for get_embeddings
        seqs_dict = {}
        for seq_id, clean_seq, temp, _ in needs_generation:
            seqs_dict[seq_id] = clean_seq

        start_time = time.time()
        results = get_embeddings(
            model, tokenizer, seqs_dict,
            per_residue=False, per_protein=True
        )
        elapsed = time.time() - start_time
        print("  Generated %d embeddings in %.1f seconds (%.2f sec/seq)" % (
            len(results['mean_representations']), elapsed,
            elapsed / max(1, len(results['mean_representations']))
        ))

        # Collect results and optionally cache
        for seq_id, clean_seq, temp, cache_path in needs_generation:
            if seq_id in results['mean_representations']:
                emb = torch.from_numpy(results['mean_representations'][seq_id]).flatten()
                embeddings_dict[seq_id] = emb

                if cache_path:
                    torch.save({
                        'label': seq_id,
                        'sequence': clean_seq,
                        'mean_representations': emb,
                    }, cache_path)

        # Free GPU/CPU memory
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Assemble in original order
    embedding_list = []
    for seq_id, seq, temp in records:
        if seq_id in embeddings_dict:
            embedding_list.append(embeddings_dict[seq_id])
        else:
            print("  WARNING: Missing embedding for %s, using zeros" % seq_id)
            embedding_list.append(torch.zeros(1024))

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
