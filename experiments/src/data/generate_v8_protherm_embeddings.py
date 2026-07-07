#!/usr/bin/env python3
"""
Generate structure-aware SaProt embeddings for ProThermDB validation sequences.
Maps available 3Di structure tokens from master datasets and outputs clean V8 representation.
"""

import os
import sys
from pathlib import Path
import torch
from Bio import SeqIO

PROJECT_ROOT = Path(__file__).resolve().parents[3]

def clean_seq(seq):
    return "".join([c for c in str(seq).upper() if c.isupper() and c.isalpha()])

def main():
    print("Generating ProThermDB V8 structure-aware embeddings dictionary...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load ProTherm validation FASTA
    protherm_fasta = PROJECT_ROOT / "new_data/prothermdb_validation.fasta"
    seqs = []
    for r in SeqIO.parse(protherm_fasta, 'fasta'):
        seqs.append(clean_seq(r.seq))
    print(f"Loaded {len(seqs)} ProThermDB validation sequences.")

    # 2. Build sequence mapping from master 3Di embeddings
    struct_data = torch.load(PROJECT_ROOT / "data/embeddings/saprot_tm_struct_embeddings.pt", map_location='cpu', weights_only=False)
    seq_to_emb = {}
    for split in ['train_tm', 'val_tm', 'test_tm']:
        if split in struct_data:
            s_list = struct_data[split]['sequences']
            e_list = struct_data[split]['embeddings']
            for s, e in zip(s_list, e_list):
                seq_to_emb[clean_seq(s)] = e

    # Also load protherm baseline embeddings for fallback if structure unavailable
    protherm_base = torch.load(PROJECT_ROOT / "data/embeddings/saprot_1.3b/protherm_embeddings.pt", map_location='cpu', weights_only=False)

    out_dict = {}
    struct_hits = 0
    for idx, s in enumerate(seqs):
        if s in seq_to_emb:
            out_dict[s] = seq_to_emb[s].clone()
            struct_hits += 1
        elif idx < len(protherm_base):
            out_dict[s] = protherm_base[idx].clone()
        else:
            out_dict[s] = torch.zeros(1280, dtype=torch.float32)

    out_path = PROJECT_ROOT / "data/embeddings/protherm_v8_struct_embeddings.pt"
    torch.save(out_dict, out_path)
    print(f"Saved {len(out_dict)} embeddings ({struct_hits} 3Di structure-aware matches) to {out_path}")

if __name__ == "__main__":
    main()
