"""
Prepare V5 multi-head data using ProtT5 embeddings.

Combines:
  - OGT data from prepared_data_full.pt (ProtT5 1024-dim, already cached)
  - Tm data from Meltome (ProtT5, just generated)
  - Test: ProThermDB (ProtT5) + OGT test set (for comparison)

Output: new_data/prepared_data_v5_prott5.pt
"""

import os
import sys
import torch
import hashlib
import numpy as np
import pandas as pd
from Bio import SeqIO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(PROJECT_ROOT, 'experiments', 'embeddings_cache')


def load_cached_embedding(seq, cache_dir=CACHE_DIR):
    """Load ProtT5 mean embedding from cache."""
    seq_trunc = seq[:1500]
    h = hashlib.sha256(seq_trunc.encode()).hexdigest()
    fpath = os.path.join(cache_dir, f'mean_{h}.pt')
    if os.path.exists(fpath):
        data = torch.load(fpath, map_location='cpu')
        emb = data.get('mean_representations', None)
        if emb is not None:
            return emb if isinstance(emb, torch.Tensor) else torch.tensor(emb)
    return None


def load_fasta_with_embeddings(fasta_path, label_dict=None, label_col='Tm'):
    """Load sequences from FASTA, retrieve cached embeddings, pair with labels."""
    embeddings = []
    labels = []
    skipped = 0

    for record in SeqIO.parse(fasta_path, 'fasta'):
        seq = str(record.seq)
        emb = load_cached_embedding(seq)
        if emb is None:
            skipped += 1
            continue

        if label_dict is not None:
            rid = record.id
            if rid in label_dict:
                labels.append(label_dict[rid])
                embeddings.append(emb)
        else:
            embeddings.append(emb)

    print(f"  {os.path.basename(fasta_path)}: {len(embeddings)} loaded, {skipped} skipped")
    if embeddings:
        return torch.stack(embeddings), torch.tensor(labels, dtype=torch.float32) if labels else None
    return None, None


def main():
    out_path = os.path.join(PROJECT_ROOT, 'new_data', 'prepared_data_v5_prott5.pt')

    # ── OGT data (reuse from prepared_data_full.pt) ──
    print("Loading OGT data from prepared_data_full.pt...")
    ogt_data = torch.load(os.path.join(PROJECT_ROOT, 'experiments', 'prepared_data_full.pt'), map_location='cpu')
    train_ogt_emb = ogt_data['train_embeddings']
    train_ogt_lbl = torch.tensor(ogt_data['train_temps'], dtype=torch.float32)
    val_ogt_emb = ogt_data['val_embeddings']
    val_ogt_lbl = torch.tensor(ogt_data['val_temps'], dtype=torch.float32)
    test_ogt_emb = ogt_data['test_embeddings']
    test_ogt_lbl = torch.tensor(ogt_data['test_temps'], dtype=torch.float32)
    print(f"  OGT train: {train_ogt_emb.shape}, val: {val_ogt_emb.shape}, test: {test_ogt_emb.shape}")

    # ── Tm data from Meltome ──
    print("\nLoading Meltome Tm data...")
    meltome_csv = os.path.join(PROJECT_ROOT, 'new_data', 'meltome_sequences.csv')
    meltome_fasta = os.path.join(PROJECT_ROOT, 'new_data', 'meltome_sequences.fasta')

    df = pd.read_csv(meltome_csv)
    # Build label dict: UniProt_ID -> Tm
    tm_dict = {}
    for _, row in df.iterrows():
        rid = str(row['UniProt_ID'])
        tm = float(row['Tm'])
        if not np.isnan(tm):
            tm_dict[rid] = tm

    print(f"  Meltome label dict: {len(tm_dict)} entries")

    # Load embeddings — FASTA IDs are "UniProt_ID|Tm", split on |
    tm_embs = []
    tm_lbls = []
    skipped = 0
    for record in SeqIO.parse(meltome_fasta, 'fasta'):
        seq = str(record.seq)
        emb = load_cached_embedding(seq)
        if emb is None:
            skipped += 1
            continue
        # FASTA ID: "A0A023PXQ4|52.4030340964799"
        uid = record.id.split('|')[0]
        if uid in tm_dict:
            tm_embs.append(emb)
            tm_lbls.append(tm_dict[uid])


    print(f"  Meltome: {len(tm_embs)} with labels, {skipped} no embedding")

    if not tm_embs:
        print("ERROR: No Tm embeddings found! Check FASTA IDs vs CSV IDs.")
        # Debug: show sample IDs
        for i, record in enumerate(SeqIO.parse(meltome_fasta, 'fasta')):
            if i < 3:
                print(f"    FASTA ID: {record.id}")
        print(f"    CSV IDs sample: {list(tm_dict.keys())[:3]}")
        sys.exit(1)

    tm_embs_t = torch.stack(tm_embs)
    tm_lbls_t = torch.tensor(tm_lbls, dtype=torch.float32)

    # Split Tm: 90% train, 5% val, 5% test
    n = len(tm_embs_t)
    indices = np.random.RandomState(42).permutation(n)
    n_train = int(0.9 * n)
    n_val = int(0.05 * n)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    # Don't use remaining as test — use ProThermDB instead

    train_tm_emb = tm_embs_t[train_idx]
    train_tm_lbl = tm_lbls_t[train_idx]
    val_tm_emb = tm_embs_t[val_idx]
    val_tm_lbl = tm_lbls_t[val_idx]

    # ── ProThermDB test set ──
    print("\nLoading ProThermDB test data...")
    protherm_csv = os.path.join(PROJECT_ROOT, 'new_data', 'prothermdb_validation.csv')
    protherm_fasta = os.path.join(PROJECT_ROOT, 'new_data', 'prothermdb_validation.fasta')

    df_p = pd.read_csv(protherm_csv)
    protherm_dict = {}
    for _, row in df_p.iterrows():
        rid = str(row['UniProt_ID'])
        tm = float(row['Tm'])
        if not np.isnan(tm):
            protherm_dict[rid] = tm

    test_tm_embs = []
    test_tm_lbls = []
    for record in SeqIO.parse(protherm_fasta, 'fasta'):
        seq = str(record.seq)
        emb = load_cached_embedding(seq)
        if emb is None:
            continue
        uid = record.id.split('|')[0]
        if uid in protherm_dict:
            test_tm_embs.append(emb)
            test_tm_lbls.append(protherm_dict[uid])

    if test_tm_embs:
        test_tm_emb = torch.stack(test_tm_embs)
        test_tm_lbl = torch.tensor(test_tm_lbls, dtype=torch.float32)
    else:
        test_tm_emb = torch.zeros(0, 1024)
        test_tm_lbl = torch.zeros(0)

    print(f"  ProThermDB test: {len(test_tm_embs)} samples")

    # ── Save ──
    print(f"\nSaving to {out_path}...")
    save_dict = {
        'train_ogt': {'embeddings': train_ogt_emb, 'labels': train_ogt_lbl},
        'val_ogt': {'embeddings': val_ogt_emb, 'labels': val_ogt_lbl},
        'test_ogt': {'embeddings': test_ogt_emb, 'labels': test_ogt_lbl},
        'train_tm': {'embeddings': train_tm_emb, 'labels': train_tm_lbl},
        'val_tm': {'embeddings': val_tm_emb, 'labels': val_tm_lbl},
        'test_tm': {'embeddings': test_tm_emb, 'labels': test_tm_lbl},
    }

    for k, v in save_dict.items():
        print(f"  {k}: emb={v['embeddings'].shape}, lbl={v['labels'].shape}")

    torch.save(save_dict, out_path)
    print(f"\nDone! Saved to {out_path}")


if __name__ == '__main__':
    main()
