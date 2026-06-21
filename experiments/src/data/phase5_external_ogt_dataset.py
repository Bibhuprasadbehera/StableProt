#!/usr/bin/env python3
"""
Phase 5: Curate External OGT Test Set (TEMPURA + BRENDA Strategy)
Downloads BRENDA FASTA and TSV, filters by taxonomic novelty against BacDive,
and runs cd-hit-2d to ensure <30% sequence identity against train_ogt.
"""

import os
import sys
import pandas as pd
import subprocess
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "new_data"
BACDIVE_CSV = PROJECT_ROOT / "data" / "ogt_labels_bacdive_corrected.csv"
TRAIN_DATA_PT = PROJECT_ROOT / "data" / "embeddings" / "prepared_data_v7_saprot1.3b_seqonly_ogt_split.pt"

TSV_FILE = DATA_DIR / "enzyme_ogt_topt.tsv"
BRENDA_FASTA = DATA_DIR / "brenda_sequences.fasta"
NOVEL_BRENDA_FASTA = DATA_DIR / "novel_brenda_ogt.fasta"
TRAIN_OGT_FASTA = DATA_DIR / "train_ogt_reference.fasta"
FINAL_FILTERED_FASTA = DATA_DIR / "external_ogt_benchmark.fasta"

def sort_fasta_by_length(fasta_path):
    print(f"Sorting {fasta_path} by sequence length descending (required by cd-hit-2d)...")
    seqs = []
    current_header = ""
    current_seq = []
    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if current_header:
                    seq_str = ''.join(current_seq)
                    seqs.append((current_header, seq_str))
                current_header = line
                current_seq = []
            else:
                current_seq.append(line)
        if current_header:
            seq_str = ''.join(current_seq)
            seqs.append((current_header, seq_str))
    
    seqs.sort(key=lambda x: len(x[1]), reverse=True)
    
    with open(fasta_path, "w") as f:
        for header, seq in seqs:
            f.write(f"{header}\n{seq}\n")
    print(f"Sorted {len(seqs)} sequences.")

def load_bacdive_organisms():
    print(f"Loading BacDive training organisms from {BACDIVE_CSV}")
    df = pd.read_csv(BACDIVE_CSV)
    organisms = set(df['organism_name'].dropna().str.lower().str.strip())
    print(f"Loaded {len(organisms)} forbidden organism names.")
    return organisms

def process_tsv_and_fasta(forbidden_organisms):
    print(f"Loading BRENDA annotations from {TSV_FILE}")
    # The columns might be: ec_num, uniprot_id, domain, species, ogt, topt
    try:
        df = pd.read_csv(TSV_FILE, sep='\t')
    except Exception as e:
        print(f"Failed to read TSV: {e}")
        return

    # Try to find species/organism column
    species_col = [c for c in df.columns if 'species' in c.lower() or 'organism' in c.lower()]
    if not species_col:
        print("Could not find species column. Columns:", df.columns)
        return
    species_col = species_col[0]

    uniprot_col = [c for c in df.columns if 'uniprot' in c.lower()][0]
    ogt_col = [c for c in df.columns if 'ogt' in c.lower()][0]

    # Filter out empty ogt
    df = df.dropna(subset=[ogt_col, uniprot_col, species_col])

    # Fix formatting: BRENDA uses underscores e.g., 'pyrococcus_furiosus', BacDive uses spaces.
    df['normalized_organism'] = df[species_col].str.replace('_', ' ').str.lower().str.strip()

    novel_df = df[~df['normalized_organism'].isin(forbidden_organisms)]
    novel_uniprots = set(novel_df[uniprot_col].unique())
    print(f"Found {len(novel_uniprots)} novel UniProt IDs with OGT after taxonomic filtering.")

    # Create mapping of uniprot to OGT
    uniprot_to_ogt = novel_df.groupby(uniprot_col)[ogt_col].first().to_dict()

    print(f"Parsing massive FASTA {BRENDA_FASTA} to extract novel sequences...")
    extracted_count = 0
    with open(BRENDA_FASTA, "r") as f_in, open(NOVEL_BRENDA_FASTA, "w") as f_out:
        current_header = ""
        current_seq = []
        is_novel = False
        
        for line in f_in:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if len(line) <= 1:
                    is_novel = False
                    continue
                    
                if is_novel and current_seq:
                    # Write previous
                    f_out.write(f"{current_header} OGT={uniprot_to_ogt[uniprot_id]}\n")
                    f_out.write(f"{''.join(current_seq)}\n")
                    extracted_count += 1
                
                # Parse new header. Example: >P0A9X9
                uniprot_id = line[1:].split()[0].split('|')[-1] # Handle >sp|P0A9X9|... if needed, or just >P0A9X9
                if uniprot_id in novel_uniprots:
                    is_novel = True
                    current_header = line
                    current_seq = []
                else:
                    is_novel = False
            else:
                if is_novel:
                    current_seq.append(line)
                    
        # Write last
        if is_novel and current_seq:
            f_out.write(f"{current_header} OGT={uniprot_to_ogt[uniprot_id]}\n")
            f_out.write(f"{''.join(current_seq)}\n")
            extracted_count += 1

    print(f"Extracted {extracted_count} taxonomically novel sequences to {NOVEL_BRENDA_FASTA}")

def create_train_reference_fasta():
    print("Exporting training sequences to FASTA for CD-HIT-2D reference...")
    data = torch.load(TRAIN_DATA_PT, map_location='cpu', weights_only=False)
    
    # We want to exclude any sequence that is in train OR val OR test, because our 941k was split.
    # Actually, if we just want it to be novel against the ENTIRE 941k dataset, we should dump all 3 splits.
    
    total = 0
    with open(TRAIN_OGT_FASTA, "w") as f:
        for split in ['train_ogt', 'val_ogt', 'test_ogt']:
            if split in data:
                seqs = data[split].get('sequences', [])
                if not seqs: continue # some splits might only have embeddings
                for i, seq in enumerate(seqs):
                    f.write(f">ref_{split}_{i}\n{seq}\n")
                    total += 1
    
    print(f"Exported {total} reference sequences to {TRAIN_OGT_FASTA}")
    return total

def run_mmseqs2():
    print("Running MMseqs2 at 30% sequence identity...")
    m8_out = DATA_DIR / "mmseqs_hits.m8"
    tmp_dir = DATA_DIR / "tmp_mmseqs"
    
    cmd = [
        "mmseqs", "easy-search",
        str(NOVEL_BRENDA_FASTA),
        str(TRAIN_OGT_FASTA),
        str(m8_out),
        str(tmp_dir),
        "--min-seq-id", "0.3",
        "-c", "0.8",
        "--cov-mode", "1",
        "--threads", "120"
    ]
    
    try:
        if not m8_out.exists() or os.path.getsize(m8_out) < 100:
            subprocess.run(cmd, check=True)
            print("MMseqs2 search completed.")
        else:
            print("MMseqs2 search results already exist. Skipping search.")
            
        # Parse m8 to find excluded IDs
        excluded_ids = set()
        with open(m8_out, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    excluded_ids.add(parts[0])
                    
        print(f"Found {len(excluded_ids)} sequences that overlap with the training set. Discarding them.")
        
        # Read novel_brenda_ogt.fasta, keep if not in excluded_ids
        survivors = []
        current_header = ""
        current_seq = []
        current_ogt = None
        
        with open(NOVEL_BRENDA_FASTA, "r") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith(">"):
                    if current_header:
                        seq_id = current_header[1:].split()[0]
                        if seq_id not in excluded_ids:
                            survivors.append((current_header, ''.join(current_seq), current_ogt))
                    
                    current_header = line
                    current_seq = []
                    # Extract OGT from header: >... OGT=37.0
                    ogt_str = line.split("OGT=")[-1]
                    try:
                        current_ogt = float(ogt_str)
                    except:
                        current_ogt = None
                else:
                    current_seq.append(line)
        
        if current_header:
            seq_id = current_header[1:].split()[0]
            if seq_id not in excluded_ids:
                survivors.append((current_header, ''.join(current_seq), current_ogt))
                
        print(f"Total surviving 100% OOD sequences: {len(survivors)}")
        
        import random
        random.shuffle(survivors)
        test_set = survivors[:5000]
        
        # Save to PyTorch dict
        pt_data = {
            'test_ogt': {
                'sequences': [s[1] for s in test_set],
                'ogt_consensus': torch.tensor([s[2] for s in test_set], dtype=torch.float32)
            }
        }
        
        torch.save(pt_data, FINAL_FILTERED_FASTA.with_suffix('.pt'))
        print(f"Saved {len(test_set)} sequences to {FINAL_FILTERED_FASTA.with_suffix('.pt')}")

    except subprocess.CalledProcessError as e:
        print(f"MMseqs2 failed: {e}")

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if not TSV_FILE.exists():
        print(f"Waiting for {TSV_FILE} to download...")
        return
        
    if not BRENDA_FASTA.exists():
        print(f"Waiting for {BRENDA_FASTA} to download...")
        return
        
    if not NOVEL_BRENDA_FASTA.exists() or os.path.getsize(NOVEL_BRENDA_FASTA) < 1000:
        forbidden = load_bacdive_organisms()
        process_tsv_and_fasta(forbidden)
    else:
        print(f"Skipping FASTA parsing, {NOVEL_BRENDA_FASTA} already exists.")
    
    # We need to make sure sequences exist in the PT file. Let's check:
    data = torch.load(TRAIN_DATA_PT, map_location='cpu', weights_only=False)
    if 'sequences' not in data['train_ogt']:
        print("WARNING: 'sequences' key not found in the PyTorch split. You may need to load the original CSV to get the sequences.")
        # Load from original CSV if needed
        pass
    else:
        # Check if reference fasta exists
        if not TRAIN_OGT_FASTA.exists() or os.path.getsize(TRAIN_OGT_FASTA) < 1000:
            create_train_reference_fasta()
            
        run_mmseqs2()

if __name__ == "__main__":
    main()
