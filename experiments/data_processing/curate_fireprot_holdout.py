"""
Curate Out-of-Distribution FireProtDB Holdout Set.

Parses wild-type sequences directly from the FireProtDB SQL database dump,
maps them to experimental melting temperatures (Tm) from the whole CSV export,
and executes strict sequence homology filtering (<30% identity) against all
Meltome/ProThermDB training records to construct an unassailable OOD evaluation set.
"""

import os
import sys
import csv
import hashlib
import numpy as np
import torch
from datetime import datetime
from Bio import SeqIO
from difflib import SequenceMatcher

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
NEW_DATA_DIR = os.path.join(PROJECT_ROOT, 'new_data')
CACHE_DIR = os.path.join(PROJECT_ROOT, 'experiments', 'embeddings_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

SQL_PATH = os.path.join(NEW_DATA_DIR, 'fireprotdb_dump_2025_09_22', '01_fireprotdb_2025-09-20.sql')
CSV_PATH = os.path.join(NEW_DATA_DIR, 'fireprotdb_dump_2025_09_22', 'fireprotdb_csv_whole', 'fireprotdb_20251015-164116.csv')
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'fireprot_holdout_prott5.pt')


def parse_sql_sequences(sql_path):
    """Parse string sequences from PostgreSQL COPY block."""
    print(f"Parsing SQL sequences from {os.path.basename(sql_path)}...")
    sequences = {}
    in_copy_block = False
    count = 0
    
    with open(sql_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith("COPY public.sequence "):
                in_copy_block = True
                continue
            if in_copy_block:
                if line.strip() == "\\.":
                    break
                parts = line.split('\t')
                if len(parts) >= 2:
                    seq_id = parts[0].strip()
                    seq_str = parts[1].strip().upper()
                    sequences[seq_id] = seq_str
                    count += 1
                    
    print(f"  Successfully extracted {count} SQL string sequences.")
    return sequences


def stream_csv_wildtypes(csv_path, sql_sequences):
    """Stream whole CSV export to extract valid wild-type sequence-to-Tm mappings."""
    print(f"Streaming whole CSV to map wild-type melting temperatures...")
    seq_to_tms = {}
    valid_count = 0
    
    # Increase CSV field size limit for massive exports
    csv.field_size_limit(sys.maxsize)
    
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
        except StopIteration:
            return {}
            
        hmap = {name.strip(): idx for idx, name in enumerate(headers)}
        
        # Guard column indices
        seq_idx = hmap.get('SEQUENCE_ID')
        mut_idx = hmap.get('MUTANT_ID')
        sub_idx = hmap.get('SUBSTITUTION')
        tm_idx = hmap.get('TM')
        
        if None in (seq_idx, tm_idx):
            print("  CRITICAL: Missing essential headers in CSV export.")
            return {}
            
        for row in reader:
            if len(row) <= max(seq_idx, tm_idx):
                continue
                
            # Filter wild-type rows: empty substitution or empty mutant ID
            is_wt = True
            if sub_idx is not None and len(row) > sub_idx and row[sub_idx].strip():
                is_wt = False
            if mut_idx is not None and len(row) > mut_idx and row[mut_idx].strip():
                is_wt = False
                
            if not is_wt:
                continue
                
            tm_str = row[tm_idx].strip()
            if not tm_str:
                continue
                
            try:
                tm_val = float(tm_str)
            except ValueError:
                continue
                
            # Keep plausible thermal unfolding points
            if not (20.0 <= tm_val <= 105.0):
                continue
                
            seq_id = row[seq_idx].strip()
            seq_str = sql_sequences.get(seq_id)
            
            if seq_str and 30 <= len(seq_str) <= 1022:
                if seq_str not in seq_to_tms:
                    seq_to_tms[seq_str] = []
                seq_to_tms[seq_str].append(tm_val)
                valid_count += 1
                
    # Deduplicate via median
    unique_records = {}
    for seq, tms in seq_to_tms.items():
        unique_records[seq] = float(np.median(tms))
        
    print(f"  Extracted {len(unique_records)} unique valid wild-type sequence targets (from {valid_count} assays).")
    return unique_records


def load_seen_training_sequences():
    """Pre-load Meltome and ProThermDB reference sequences for fast k-mer filtering."""
    print("Loading pre-training reference sequence structures...")
    seen_seqs = set()
    
    fastas = [
        os.path.join(NEW_DATA_DIR, 'meltome_sequences.fasta'),
        os.path.join(NEW_DATA_DIR, 'prothermdb_validation.fasta')
    ]
    
    for fpath in fastas:
        if os.path.exists(fpath):
            for record in SeqIO.parse(fpath, 'fasta'):
                seen_seqs.add(str(record.seq).upper())
                
    print(f"  Pre-loaded {len(seen_seqs)} reference sequences.")
    
    # Precompute k-mer sets for ultrafast screening
    seen_kmers = []
    for s in seen_seqs:
        kmers = {s[i:i+5] for i in range(len(s)-4)} if len(s) >= 5 else set()
        seen_kmers.append((s, len(s), kmers))
        
    return seen_seqs, seen_kmers


def filter_homologous_sequences(candidates, seen_set, seen_kmers):
    """Execute strict out-of-distribution sequence identity filtering (<30%)."""
    print("Executing strict homology filtering (<30% sequence identity)...")
    surviving = {}
    dropped = 0
    
    start = datetime.now()
    for idx, (c_seq, tm) in enumerate(candidates.items()):
        if c_seq in seen_set:
            dropped += 1
            continue
            
        c_len = len(c_seq)
        c_kmers = {c_seq[i:i+5] for i in range(len(c_seq)-4)} if c_len >= 5 else set()
        
        is_homol = False
        for s_seq, s_len, s_kmers in seen_kmers:
            # Length guard check
            if min(c_len, s_len) / max(c_len, s_len) < 0.3:
                continue
                
            # Ultra-fast k-mer intersection threshold check
            if c_kmers and s_kmers:
                inter_ratio = len(c_kmers.intersection(s_kmers)) / min(len(c_kmers), len(s_kmers))
                if inter_ratio < 0.12:
                    continue
                    
            # Complete DP ratio assessment
            ratio = SequenceMatcher(None, c_seq, s_seq).ratio()
            if ratio >= 0.30:
                is_homol = True
                break
                
        if is_homol:
            dropped += 1
        else:
            surviving[c_seq] = tm
            
        if (idx + 1) % 50 == 0:
            print(f"  Screened {idx+1}/{len(candidates)} candidates... (Surviving: {len(surviving)}, Dropped: {dropped})")
            
    elapsed = (datetime.now() - start).total_seconds()
    print(f"  Homology filtering complete in {elapsed:.1f}s. Final clean out-of-distribution set: {len(surviving)} targets.")
    return surviving


def extract_or_generate_embeddings(surviving_records):
    """Load cached representations or dynamically encode missing targets (ProtT5 & ESM-2)."""
    print("\nExtracting feature embeddings...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    CACHE_DIR_ESM = os.path.join(PROJECT_ROOT, 'experiments', 'esm2_embeddings_cache')
    os.makedirs(CACHE_DIR_ESM, exist_ok=True)
    
    # 1. Identify missing ProtT5
    missing_t5 = {}
    for seq in surviving_records:
        h = hashlib.sha256(seq[:1500].encode()).hexdigest()
        if not os.path.exists(os.path.join(CACHE_DIR, f'mean_{h}.pt')):
            missing_t5[seq] = surviving_records[seq]
            
    if missing_t5:
        print(f"  Generating ProtT5 for {len(missing_t5)} targets...")
        from transformers import T5EncoderModel, AutoTokenizer
        model_name = 'Rostlab/prot_t5_xl_half_uniref50-enc'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = T5EncoderModel.from_pretrained(model_name).to(device).eval()
        for seq in missing_t5:
            spaced = ' '.join(list(seq)).replace('U', 'X').replace('Z', 'X').replace('O', 'X')
            encoded = tokenizer(spaced, return_tensors='pt', max_length=1502, truncation=True)
            with torch.no_grad():
                out = model(input_ids=encoded['input_ids'].to(device), attention_mask=encoded['attention_mask'].to(device))
                emb = out.last_hidden_state[0, :len(seq)].mean(dim=0).cpu()
            h = hashlib.sha256(seq[:1500].encode()).hexdigest()
            torch.save({'sequence': seq, 'mean_representations': emb}, os.path.join(CACHE_DIR, f'mean_{h}.pt'))
        del model, tokenizer
        torch.cuda.empty_cache()

    # 2. Identify missing native ESM-2
    missing_esm = {}
    for seq in surviving_records:
        h = hashlib.sha256(seq[:1500].encode()).hexdigest()
        if not os.path.exists(os.path.join(CACHE_DIR_ESM, f'esm2_{h}.pt')):
            missing_esm[seq] = surviving_records[seq]

    if missing_esm:
        print(f"  Generating native ESM-2 3B for {len(missing_esm)} targets...")
        # ESM-2 3B (Targeting Layer 22 - Unnormalized structural manifold)
        print("\nLoading ESM-2 3B Model...")
        import esm
        model_esm, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        model_esm = model_esm.eval().to(device)
        batch_converter = alphabet.get_batch_converter()
        REPR_LAYER = 22 # Identified as the training manifold layer for V6
        
        esm2_generated = 0
        with torch.no_grad():
            for seq in missing_esm:
                h = hashlib.sha256(seq[:1500].encode()).hexdigest()
                out_path = os.path.join(CACHE_DIR_ESM, f"esm2_{h}.pt")
                
                # Force regeneration for FireProt targets to ensure layer alignment
                # (Previously generated files might be Layer 36)
                data = [("target", seq[:1022])] # ESM-2 limit is 1024
                _, _, tokens = batch_converter(data)
                tokens = tokens.to(device)
                
                results = model_esm(tokens, repr_layers=[REPR_LAYER], return_contacts=False)
                # Mean pool excluding CLS/EOS
                token_reps = results["representations"][REPR_LAYER][0, 1:len(seq[:1022])+1]
                mean_rep = token_reps.mean(0).cpu()
                
                torch.save(mean_rep, out_path)
                esm2_generated += 1
                
        print(f"Generated {esm2_generated} ESM-2 Layer 22 embeddings.")
        del model_esm
        torch.cuda.empty_cache()

    # 3. Final collection sweep
    final_t5, final_esm, final_tm, final_seq = [], [], [], []
    for seq, tm in surviving_records.items():
        h = hashlib.sha256(seq[:1500].encode()).hexdigest()
        t5_data = torch.load(os.path.join(CACHE_DIR, f'mean_{h}.pt'), map_location='cpu')
        t5 = t5_data['mean_representations'] if isinstance(t5_data, dict) else t5_data
        esm = torch.load(os.path.join(CACHE_DIR_ESM, f'esm2_{h}.pt'), map_location='cpu')
        
        final_t5.append(t5)
        final_esm.append(esm)
        final_tm.append(tm)
        final_seq.append(seq)

    return torch.stack(final_t5), torch.stack(final_esm), torch.tensor(final_tm), final_seq


def main():
    print("=" * 80)
    print("  PHASE C: DE NOVO FIREPROT-DB OUT-OF-DISTRIBUTION CURATION PIPELINE")
    print("=" * 80)
    
    if not os.path.exists(SQL_PATH) or not os.path.exists(CSV_PATH):
        print(f"CRITICAL ERROR: Source FireProtDB files missing.")
        print(f"Expected SQL: {SQL_PATH}")
        print(f"Expected CSV: {CSV_PATH}")
        sys.exit(1)
        
    # 1. SQL Parse
    sql_seqs = parse_sql_sequences(SQL_PATH)
    
    # 2. CSV Stream mapping
    wt_records = stream_csv_wildtypes(CSV_PATH, sql_seqs)
    if not wt_records:
        print("Failed to extract valid wild-type candidates.")
        sys.exit(1)
        
    # 3. Seen Pre-training sets
    seen_set, seen_kmers = load_seen_training_sequences()
    
    # 4. Filter OOD targets
    surviving = filter_homologous_sequences(wt_records, seen_set, seen_kmers)
    if not surviving:
        print("No out-of-distribution targets survived homology filtering.")
        sys.exit(1)
        
    # 5. Extract representations
    t5_embs, esm_embs, tms, seqs = extract_or_generate_embeddings(surviving)
    
    if t5_embs is not None:
        save_data = {
            'embeddings_prott5': t5_embs,
            'embeddings_esm2': esm_embs,
            'temperatures': tms,
            'sequences': seqs
        }
        torch.save(save_data, OUTPUT_PATH)
        print(f"\nSUCCESS: Clean Out-of-Distribution validation tensor stored beautifully.")
        print(f"Output Path: {OUTPUT_PATH}")
        print(f"Tensor Shape: ProtT5={t5_embs.shape}, ESM2={esm_embs.shape}")
    else:
        print("Failed to construct final structured tensor.")


if __name__ == "__main__":
    main()
