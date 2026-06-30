#!/usr/bin/env python3
"""
Step 2b: Foldseek 3Di Token Extraction + SaProt Structure-Aware Embedding

After all ESMFold PDBs arrive from HPC, this script:
1. Runs Foldseek on all PDBs → 3Di structural tokens
2. Combines sequence + 3Di tokens into SA format for SaProt
3. Extracts SaProt 1.3B embeddings with SA tokens

Usage:
    python phase2b_structure_aware_embeddings.py [--foldseek-only] [--embed-only]
"""

import argparse
import gc
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PDB_DIR = PROJECT_ROOT / "data" / "structures" / "complete_pdbs"
FOLDSEEK_DIR = PROJECT_ROOT / "data" / "structures" / "foldseek_output"
V7_DATA = PROJECT_ROOT / "data" / "embeddings" / "prepared_data_v7_saprot1.3b_seqonly.pt"
SA_OUTPUT = PROJECT_ROOT / "data" / "embeddings" / "saprot_tm_struct_embeddings.pt"

# SaProt uses this alphabet for structural tokens
STRUC_ALPHABET = "pynwrqhgdlvtmfsaeikc#"  # 20 structural + 1 mask


def run_foldseek(pdb_dir, output_dir):
    """Run Foldseek on all PDBs to extract 3Di structural tokens.
    
    Creates a Foldseek database, then extracts 3Di sequences.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path = output_dir / "tm_db"
    tokens_path = output_dir / "3di_tokens.tsv"

    if tokens_path.exists():
        print(f"  3Di tokens already exist: {tokens_path}")
        return tokens_path

    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    print(f"  PDB files found: {len(pdb_files)}")
    if len(pdb_files) == 0:
        raise RuntimeError(f"No PDB files found in {pdb_dir}")

    # Step 1: Create Foldseek database from PDB directory
    print("  Creating Foldseek database...")
    t0 = time.time()
    cmd_create = [
        "foldseek", "createdb",
        str(pdb_dir),
        str(db_path),
    ]
    result = subprocess.run(cmd_create, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-500:]}")
        raise RuntimeError("Foldseek createdb failed")
    print(f"  Database created in {time.time()-t0:.1f}s")

    # Step 2: Extract 3Di sequences (structural tokens)
    print("  Extracting 3Di structural tokens...")
    t0 = time.time()

    # Foldseek stores 3Di in _ss database
    # Use convert2fasta or lndb + result2flat to extract
    cmd_extract = [
        "foldseek", "convert2fasta",
        str(db_path) + "_ss",
        str(output_dir / "3di_sequences.fasta"),
    ]
    result = subprocess.run(cmd_extract, capture_output=True, text=True)
    if result.returncode != 0:
        # Fallback: extract using lndb
        print(f"  convert2fasta failed, trying lndb approach...")
        cmd_lndb = [
            "foldseek", "lndb",
            str(db_path) + "_h",
            str(db_path) + "_ss_h",
        ]
        subprocess.run(cmd_lndb, capture_output=True, text=True)
        cmd_extract2 = [
            "foldseek", "convert2fasta",
            str(db_path) + "_ss",
            str(output_dir / "3di_sequences.fasta"),
        ]
        result = subprocess.run(cmd_extract2, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Foldseek 3Di extraction failed: {result.stderr[-500:]}")

    print(f"  3Di tokens extracted in {time.time()-t0:.1f}s")

    # Parse 3Di FASTA into TSV: pdb_id \t 3di_sequence
    fasta_path = output_dir / "3di_sequences.fasta"
    with open(fasta_path) as f, open(tokens_path, 'w') as out:
        current_id = None
        current_seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    out.write(f"{current_id}\t{''.join(current_seq)}\n")
                current_id = line[1:].split()[0]
                # Strip .pdb extension if present
                if current_id.endswith(".pdb"):
                    current_id = current_id[:-4]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id is not None:
            out.write(f"{current_id}\t{''.join(current_seq)}\n")

    # Count entries
    n = sum(1 for _ in open(tokens_path))
    print(f"  Saved {n} 3Di token sequences to {tokens_path}")

    return tokens_path


def load_3di_tokens(tokens_path):
    """Load 3Di tokens from TSV: {pdb_id: 3di_sequence}."""
    tokens = {}
    with open(tokens_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                tokens[parts[0]] = parts[1]
    return tokens


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


def build_sa_sequences(v7_data, tokens_3di):
    """Combine amino acid sequences with 3Di tokens into SaProt SA format.
    
    SaProt SA format interleaves: A1 S1 A2 S2 A3 S3 ...
    where Ai = amino acid, Si = structural token (lowercase 3Di)
    
    If 3Di token is missing, use '#' (mask token).
    """
    from concurrent.futures import ProcessPoolExecutor

    print("  Mapping PDB structures to amino acid sequences...")
    all_pdbs = list(PDB_DIR.glob('*.pdb'))
    with ProcessPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_get_pdb_seq, all_pdbs))

    seq_to_3di = {}
    for stem, aa_seq in results:
        if stem in tokens_3di and len(aa_seq) == len(tokens_3di[stem]):
            seq_to_3di[aa_seq] = tokens_3di[stem]

    sa_sequences = {}

    for split in ['train_tm', 'val_tm', 'test_tm']:
        sequences = v7_data[split]['sequences']
        ids = v7_data[split]['ids']
        sa_seqs = []

        matched = 0
        masked = 0

        for i, (seq, seq_id) in enumerate(zip(sequences, ids)):
            aa_seq = str(seq).upper()
            pdb_id = f"{split}_{i}"
            tokens_3d = tokens_3di.get(pdb_id, None)

            # Check direct ID match first, then fallback to sequence match
            if tokens_3d is None or len(tokens_3d) != len(aa_seq):
                tokens_3d = seq_to_3di.get(aa_seq, None)

            if tokens_3d is not None and len(tokens_3d) == len(aa_seq):
                # Interleave: A1s1A2s2...
                sa_seq = ''.join(
                    aa + tok.lower() for aa, tok in zip(aa_seq, tokens_3d)
                )
                matched += 1
            else:
                # Fallback: mask all structure tokens
                sa_seq = ''.join(aa + '#' for aa in aa_seq)
                masked += 1

            sa_seqs.append(sa_seq)

        sa_sequences[split] = sa_seqs
        pct = 100 * matched / (matched + masked) if (matched + masked) > 0 else 0
        print(f"  {split}: {matched} SA + {masked} masked ({pct:.1f}% structure-aware)")

    return sa_sequences


def extract_saprot_embeddings(sa_sequences, v7_data, batch_size=4):
    """Extract SaProt 1.3B embeddings from SA-format sequences."""
    from transformers import EsmTokenizer, EsmModel

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Loading SaProt 1.3B model on {device}...")

    model_name = "westlake-repl/SaProt_1.3B_AFDB_OMG_NCBI"
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    print(f"  Model loaded. Hidden size: {model.config.hidden_size}")

    result = {}
    for split in ['train_tm', 'val_tm', 'test_tm']:
        seqs = sa_sequences[split]
        print(f"\n  Processing {split}: {len(seqs)} sequences...")

        all_embeddings = []
        for i in tqdm(range(0, len(seqs), batch_size), desc=f"  {split}"):
            batch = seqs[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True,
                             truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                # Mean pool over sequence length (excluding padding)
                mask = inputs['attention_mask'].unsqueeze(-1)
                pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
                all_embeddings.append(pooled.cpu())

            if (i // batch_size) % 100 == 0:
                torch.cuda.empty_cache()

        embeddings = torch.cat(all_embeddings, dim=0)
        print(f"  {split} embeddings: {embeddings.shape}")
        result[split] = embeddings

    return result


def merge_and_save(embeddings, v7_data):
    """Replace embeddings in V7 data and save."""
    for split in ['train_tm', 'val_tm', 'test_tm']:
        v7_data[split]['embeddings'] = embeddings[split]
        print(f"  {split}: {embeddings[split].shape}")

    # Keep OGT embeddings as-is (sequence-only)
    print(f"  train_ogt: {v7_data['train_ogt']['embeddings'].shape} (sequence-only, unchanged)")

    torch.save(v7_data, SA_OUTPUT)
    print(f"\n  Saved structure-aware data: {SA_OUTPUT}")
    print(f"  File size: {SA_OUTPUT.stat().st_size / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Phase 2b: Structure-Aware SaProt Embeddings")
    parser.add_argument("--foldseek-only", action="store_true", help="Only run Foldseek, skip embedding")
    parser.add_argument("--embed-only", action="store_true", help="Skip Foldseek, only extract embeddings")
    parser.add_argument("--batch-size", type=int, default=4, help="SaProt batch size")
    args = parser.parse_args()

    print("Phase 2b: Structure-Aware SaProt Embeddings")
    print("=" * 60)

    # Check PDB directory
    pdb_count = len(list(PDB_DIR.glob("*.pdb")))
    print(f"  PDB files available: {pdb_count}")

    # Load V7 data for sequence info
    print("  Loading V7 data...")
    v7_data = torch.load(V7_DATA, map_location='cpu', weights_only=False)
    total_tm = sum(len(v7_data[s]['sequences']) for s in ['train_tm', 'val_tm', 'test_tm'])
    print(f"  Total Tm sequences: {total_tm}")
    print(f"  PDB coverage: {pdb_count}/{total_tm} ({100*pdb_count/total_tm:.1f}%)")

    if pdb_count < total_tm * 0.95:
        print(f"\n  WARNING: Only {100*pdb_count/total_tm:.1f}% PDB coverage.")
        print(f"  Missing {total_tm - pdb_count} PDBs. Wait for HPC job to finish.")
        print(f"  Continuing anyway — missing sequences will use #mask fallback.")

    if not args.embed_only:
        # Step 1: Foldseek
        print("\n--- Step 1: Foldseek 3Di Token Extraction ---")
        tokens_path = run_foldseek(PDB_DIR, FOLDSEEK_DIR)
    else:
        tokens_path = FOLDSEEK_DIR / "3di_tokens.tsv"

    if args.foldseek_only:
        print("\n  --foldseek-only: stopping after Foldseek.")
        return

    # Step 2: Build SA sequences
    print("\n--- Step 2: Build SA Sequences ---")
    padded_tokens_path = FOLDSEEK_DIR / "3di_tokens_padded.tsv"
    if padded_tokens_path.exists():
        print(f"  Using verified padded 3Di tokens: {padded_tokens_path}")
        tokens_path = padded_tokens_path
    tokens_3di = load_3di_tokens(tokens_path)
    print(f"  Loaded {len(tokens_3di)} 3Di token sequences")
    sa_sequences = build_sa_sequences(v7_data, tokens_3di)

    # Step 3: Extract embeddings
    print("\n--- Step 3: SaProt 1.3B Embedding Extraction ---")
    embeddings = extract_saprot_embeddings(sa_sequences, v7_data, batch_size=args.batch_size)

    # Step 4: Save
    print("\n--- Step 4: Merge and Save ---")
    merge_and_save(embeddings, v7_data)


if __name__ == "__main__":
    main()
