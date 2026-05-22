import os
import hashlib
from Bio import SeqIO

def get_seq_hash(seq, max_len=None):
    if max_len:
        seq = seq[:max_len]
    return hashlib.md5(seq.encode()).hexdigest()

cache_dir = "experiments/esm2_embeddings_cache"
tm_fasta = "new_data/meltome_sequences.fasta"

print(f"Checking first 100 sequences from {tm_fasta}")
found_trunc = 0
found_full = 0
for i, record in enumerate(SeqIO.parse(tm_fasta, "fasta")):
    if i >= 100: break
    seq = str(record.seq)
    
    h_trunc = get_seq_hash(seq, 1500)
    h_full = get_seq_hash(seq, None)
    
    exists_trunc = os.path.exists(os.path.join(cache_dir, f"esm2_{h_trunc}.pt"))
    exists_full = os.path.exists(os.path.join(cache_dir, f"esm2_{h_full}.pt"))
    
    if exists_trunc: found_trunc += 1
    if exists_full: found_full += 1

print(f"\nSummary (100 seqs):")
print(f"Found with 1500-truncation: {found_trunc}")
print(f"Found with full-length: {found_full}")
