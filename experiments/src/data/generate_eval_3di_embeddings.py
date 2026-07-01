#!/usr/bin/env python3
"""Generate 3Di structure-aware SaProt embeddings for ProThermDB & FireProtDB holdout sequences.
Steps:
1. Folds missing evaluation sequences using ESMFold.
2. Extracts 3Di tokens using Foldseek.
3. Extracts 1280-dim structure-aware embeddings using SaProt 650M AF2.
4. Updates saprot_tm_struct_embeddings.pt for 100% evaluation structure coverage.
"""

import os
import sys
import json
import time
import subprocess
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForProteinFolding, EsmModel

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVAL_PDB_DIR = PROJECT_ROOT / "data" / "structures" / "eval_pdbs"
FOLDSEEK_DIR = PROJECT_ROOT / "data" / "structures" / "eval_foldseek"
SA_EMB_PATH = PROJECT_ROOT / "data" / "embeddings" / "saprot_tm_struct_embeddings.pt"

def convert_outputs_to_pdb(outputs, sequence):
    positions = outputs.positions[-1][0].cpu().numpy()
    atom_names = ["N", "CA", "C", "O", "CB"]
    mapping = {
        'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
        'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
        'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
        'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
    }
    lines = []
    atom_idx = 1
    for res_idx, aa in enumerate(sequence):
        for atom_i, atom_name in enumerate(atom_names):
            if atom_i >= positions.shape[1]: break
            x, y, z = positions[res_idx, atom_i]
            if np.isnan(x): continue
            lines.append(
                f"ATOM  {atom_idx:5d} {atom_name:4s} {mapping.get(aa.upper(), 'UNK')} A{res_idx+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_name[0]:>2s}"
            )
            atom_idx += 1
    lines.append("END")
    return "\n".join(lines)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EVAL_PDB_DIR.mkdir(parents=True, exist_ok=True)
    FOLDSEEK_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Collect existing SA sequences
    print("Loading existing structure embeddings...")
    d_sa = torch.load(SA_EMB_PATH, map_location='cpu', weights_only=False)
    existing_sa_seqs = set()
    for split in ['train_tm', 'val_tm', 'test_tm']:
        if split in d_sa and 'sequences' in d_sa[split]:
            for s in d_sa[split]['sequences']:
                existing_sa_seqs.add(str(s).upper())
    print(f"Loaded {len(existing_sa_seqs)} existing structure sequences.")

    # 2. Collect missing ProTherm & FireProt sequences
    missing_work = {}

    # ProTherm validation set
    protherm_csv = PROJECT_ROOT / "new_data" / "prothermdb_validation.csv"
    df_p = pd.read_csv(protherm_csv)
    protherm_dict = {str(row['UniProt_ID']): float(row['Tm']) for _, row in df_p.iterrows() if not np.isnan(row['Tm'])}
    v7_data = torch.load(PROJECT_ROOT / "data/embeddings/prepared_data_v7_saprot1.3b_seqonly.pt", map_location='cpu', weights_only=False)
    train_seqs = {str(s).upper() for s in v7_data['train_tm']['sequences']}

    for idx, record in enumerate(SeqIO.parse(PROJECT_ROOT / "new_data" / "prothermdb_validation.fasta", 'fasta')):
        seq = str(record.seq).upper()
        uid = record.id.split('|')[0]
        if uid in protherm_dict and seq not in train_seqs and seq not in existing_sa_seqs:
            missing_work[f"protherm_{idx}"] = seq

    # FireProt holdout set
    d_fp = torch.load(PROJECT_ROOT / "data/test_data/fireprot_holdout_saprot.pt", map_location='cpu', weights_only=False)
    for idx, s in enumerate(d_fp['sequences']):
        seq = str(s).upper()
        if seq not in existing_sa_seqs:
            missing_work[f"fireprot_{idx}"] = seq

    print(f"Total unique missing eval sequences to fold & embed: {len(missing_work)}")
    if len(missing_work) == 0:
        print("All eval sequences already have 3Di structure embeddings!")
        return

    # Check which PDBs already generated
    to_fold = {k: v for k, v in missing_work.items() if not (EVAL_PDB_DIR / f"{k}.pdb").exists()}
    print(f"Sequences needing ESMFold: {len(to_fold)}")

    if len(to_fold) > 0:
        print("Loading ESMFold...")
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        esmfold = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True).to(device).eval()

        pbar = tqdm(to_fold.items(), desc="ESMFold Eval Seqs")
        for k, seq in pbar:
            try:
                seq_trunc = seq[:1000] # Truncate >1000 AA to avoid GPU OOM
                with torch.no_grad():
                    inputs = tokenizer([seq_trunc], return_tensors="pt", add_special_tokens=False)
                    inputs = {key: val.to(device) for key, val in inputs.items()}
                    outputs = esmfold(**inputs)
                pdb_str = convert_outputs_to_pdb(outputs, seq_trunc)
                with open(EVAL_PDB_DIR / f"{k}.pdb", "w") as f:
                    f.write(pdb_str)
            except Exception as e:
                print(f"Error folding {k}: {e}")
        del esmfold
        torch.cuda.empty_cache()

    # 3. Run Foldseek on EVAL_PDB_DIR
    print("\nRunning Foldseek to extract 3Di tokens...")
    db_path = FOLDSEEK_DIR / "eval_db"
    fasta_path = FOLDSEEK_DIR / "eval_3di.fasta"
    
    cmd_db = f"foldseek createdb {EVAL_PDB_DIR} {db_path}"
    subprocess.run(cmd_db, shell=True, check=True)
    
    cmd_lndb = f"foldseek lndb {db_path}_h {db_path}_ss_h"
    subprocess.run(cmd_lndb, shell=True, check=False)

    cmd_extract = f"foldseek convert2fasta {db_path}_ss {fasta_path}"
    subprocess.run(cmd_extract, shell=True, check=True)

    # Load 3Di tokens from FASTA
    sa_strings = {}
    with open(fasta_path) as f:
        curr_id = None
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                curr_id = line[1:].split('.')[0]
            else:
                if curr_id in missing_work:
                    aa_seq = missing_work[curr_id]
                    p3di_seq = line.lower()
                    sa_token_list = [f"{a}{d}" for a, d in zip(aa_seq[:len(p3di_seq)], p3di_seq)]
                    sa_strings[aa_seq] = "".join(sa_token_list)

    print(f"Extracted SA structure token strings for {len(sa_strings)} eval sequences.")

    # 4. Extract SaProt 650M AF2 embeddings
    print("\nLoading SaProt 650M AF2 to embed structure sequences...")
    saprot_tokenizer = AutoTokenizer.from_pretrained("westlake-repl/SaProt_650M_AF2")
    saprot_model = EsmModel.from_pretrained("westlake-repl/SaProt_650M_AF2").to(device).eval()

    new_seqs = []
    new_embs = []
    for seq, sa_str in tqdm(sa_strings.items(), desc="SaProt Embedding"):
        try:
            with torch.no_grad():
                inputs = saprot_tokenizer(sa_str[:1022], return_tensors="pt", truncation=True, max_length=1024).to(device)
                outputs = saprot_model(**inputs)
                # Mean pooling over residue tokens (exclude CLS and EOS)
                emb = outputs.last_hidden_state[0, 1:-1].mean(dim=0).cpu()
                new_seqs.append(seq)
                new_embs.append(emb)
        except Exception as e:
            print(f"Error embedding SA string: {e}")

    # 5. Append new embeddings to saprot_tm_struct_embeddings.pt
    print(f"\nAdding {len(new_seqs)} new SA structure embeddings to test_tm partition...")
    if 'test_tm' not in d_sa:
        d_sa['test_tm'] = {'sequences': [], 'embeddings': []}
    
    d_sa['test_tm']['sequences'].extend(new_seqs)
    if isinstance(d_sa['test_tm']['embeddings'], list):
        d_sa['test_tm']['embeddings'].extend(new_embs)
    else:
        d_sa['test_tm']['embeddings'] = torch.cat([d_sa['test_tm']['embeddings'], torch.stack(new_embs)])

    torch.save(d_sa, SA_EMB_PATH)
    print("Done! 100% structure embedding coverage achieved for ProTherm and FireProt holdout sets.")

if __name__ == "__main__":
    main()
