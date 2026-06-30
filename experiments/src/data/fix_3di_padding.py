#!/usr/bin/env python3
"""
Step 1: Fix 3Di Token Padding for 100% Structure-Aware Alignment.

Loads:
1. prepared_data_v7_saprot1.3b_seqonly.pt (or v4)
2. PDB sequences & Foldseek 3Di tokens (3di_tokens.tsv)

Matches sequences by exact match or prefix match (for 900-AA capped PDBs).
Pads missing tail 3Di tokens with '#' (mask token) or truncates extra tokens
so that len(3Di) == len(sequence) exactly for 100% of sequences.
Saves: data/structures/foldseek_output/3di_tokens_padded.tsv
"""
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PDB_DIR = PROJECT_ROOT / "data" / "structures" / "complete_pdbs"
FOLDSEEK_DIR = PROJECT_ROOT / "data" / "structures" / "foldseek_output"
TOKENS_PATH = FOLDSEEK_DIR / "3di_tokens.tsv"
OUT_TOKENS_PATH = FOLDSEEK_DIR / "3di_tokens_padded.tsv"
V7_DATA = PROJECT_ROOT / "data" / "embeddings" / "prepared_data_v7_saprot1.3b_seqonly.pt"
if not V7_DATA.exists():
    V7_DATA = PROJECT_ROOT / "data" / "embeddings" / "prepared_data_v4_saprot.pt"

D3TO1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
         'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
         'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
         'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}


def _get_pdb_seq(pdb_path):
    seq = []
    seen = set()
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith('ATOM ') and line[12:16].strip() == 'CA':
                    r = line[22:27]
                    if r not in seen:
                        seen.add(r)
                        seq.append(D3TO1.get(line[17:20].strip(), 'X'))
    except Exception:
        pass
    return pdb_path.stem, ''.join(seq)


def main():
    print(f"Loading dataset from {V7_DATA}...")
    data = torch.load(V7_DATA, map_location='cpu', weights_only=False)

    print(f"Loading raw 3Di tokens from {TOKENS_PATH}...")
    raw_3di = {}
    with open(TOKENS_PATH) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                raw_3di[parts[0]] = parts[1]

    print(f"Loaded {len(raw_3di)} raw 3Di entries.")

    print("Extracting amino acid sequences from PDB files...")
    all_pdbs = sorted(list(PDB_DIR.glob('*.pdb')))
    with ProcessPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_get_pdb_seq, all_pdbs))

    # Map stem -> pdb_aa_seq
    stem_to_pdb_aa = {}
    for stem, aa_seq in results:
        stem_to_pdb_aa[stem] = aa_seq

    # Map pdb_aa_seq -> 3di_seq
    exact_aa_to_3di = {}
    for stem, aa_seq in results:
        if stem in raw_3di and len(aa_seq) == len(raw_3di[stem]):
            exact_aa_to_3di[aa_seq] = raw_3di[stem]

    # Also build a list of (pdb_aa_seq, 3di_seq) sorted by length descending for prefix matching
    prefix_candidates = sorted(exact_aa_to_3di.items(), key=lambda x: len(x[0]), reverse=True)

    print("\nMatching and padding 3Di tokens across dataset splits...")
    total_seqs = 0
    exact_matches = 0
    padded_matches = 0
    truncated_matches = 0
    fallback_masked = 0

    aligned_tokens = {}  # seq_id_in_v7 -> padded_3di

    for split in ['train_tm', 'val_tm', 'test_tm']:
        if split not in data:
            continue
        sequences = data[split]['sequences']
        ids = data[split].get('ids', [f"{split}_{i}" for i in range(len(sequences))])

        for i, (seq, seq_id) in enumerate(zip(sequences, ids)):
            total_seqs += 1
            aa_seq = str(seq).upper()
            target_len = len(aa_seq)

            # 1. Try exact AA match
            if aa_seq in exact_aa_to_3di:
                toks = exact_aa_to_3di[aa_seq]
                if len(toks) == target_len:
                    aligned_tokens[f"{split}_{i}"] = toks
                    exact_matches += 1
                    continue

            # 2. Try stem lookup if available
            stem_toks = raw_3di.get(f"{split}_{i}", None)
            if stem_toks is not None:
                if len(stem_toks) == target_len:
                    aligned_tokens[f"{split}_{i}"] = stem_toks
                    exact_matches += 1
                    continue
                elif len(stem_toks) < target_len:
                    # Pad
                    padded = stem_toks + '#' * (target_len - len(stem_toks))
                    aligned_tokens[f"{split}_{i}"] = padded
                    padded_matches += 1
                    continue
                else:
                    # Truncate
                    trunc = stem_toks[:target_len]
                    aligned_tokens[f"{split}_{i}"] = trunc
                    truncated_matches += 1
                    continue

            # 3. Try prefix match in exact_aa_to_3di (e.g., PDB capped at 900 AA)
            matched_prefix = False
            # Check prefix of length 900 or 1000 specifically first for speed
            for prefix_len in [1000, 900]:
                if target_len > prefix_len:
                    sub_aa = aa_seq[:prefix_len]
                    if sub_aa in exact_aa_to_3di:
                        base_3di = exact_aa_to_3di[sub_aa]
                        padded = base_3di + '#' * (target_len - len(base_3di))
                        aligned_tokens[f"{split}_{i}"] = padded
                        padded_matches += 1
                        matched_prefix = True
                        break

            if matched_prefix:
                continue

            # General prefix matching fallback
            for cand_aa, cand_3di in prefix_candidates:
                if target_len > len(cand_aa) and aa_seq.startswith(cand_aa):
                    padded = cand_3di + '#' * (target_len - len(cand_3di))
                    aligned_tokens[f"{split}_{i}"] = padded
                    padded_matches += 1
                    matched_prefix = True
                    break
                elif target_len < len(cand_aa) and cand_aa.startswith(aa_seq):
                    trunc = cand_3di[:target_len]
                    aligned_tokens[f"{split}_{i}"] = trunc
                    truncated_matches += 1
                    matched_prefix = True
                    break

            if not matched_prefix:
                # Complete fallback
                aligned_tokens[f"{split}_{i}"] = '#' * target_len
                fallback_masked += 1

    print("\n--- 3DI PADDING & ALIGNMENT AUDIT ---")
    print(f"Total sequences evaluated: {total_seqs}")
    print(f"Exact 1:1 matches:         {exact_matches} ({100*exact_matches/total_seqs:.2f}%)")
    print(f"Padded (tail residues):    {padded_matches} ({100*padded_matches/total_seqs:.2f}%)")
    print(f"Truncated (extra PDB):     {truncated_matches} ({100*truncated_matches/total_seqs:.2f}%)")
    print(f"Complete masked fallback:  {fallback_masked} ({100*fallback_masked/total_seqs:.2f}%)")

    # Save padded tokens
    print(f"\nSaving 100% aligned 3Di tokens to {OUT_TOKENS_PATH}...")
    with open(OUT_TOKENS_PATH, 'w') as out:
        for k, v in aligned_tokens.items():
            out.write(f"{k}\t{v}\n")
    print("Done!")


if __name__ == "__main__":
    main()
