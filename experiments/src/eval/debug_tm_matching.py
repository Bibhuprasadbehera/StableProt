import os
import hashlib
from Bio import SeqIO
import torch

def get_seq_hash(seq, max_len=1500):
    # ESM-2 generation uses truncation to 1500 before hashing
    truncated_seq = seq[:max_len]
    return hashlib.md5(truncated_seq.encode()).hexdigest()

cache_dir = "experiments/esm2_embeddings_cache"
tm_fasta = "new_data/meltome_sequences.fasta"

print(f"Checking first 5 sequences from {tm_fasta}")
found = 0
for i, record in enumerate(SeqIO.parse(tm_fasta, "fasta")):
    if i >= 5: break
    seq = str(record.seq)
    seq_hash = get_seq_hash(seq)
    emb_path = os.path.join(cache_dir, f"esm2_{seq_hash}.pt")
    exists = os.path.exists(emb_path)
    print(f"Seq {i}: Hash={seq_hash}, Exists={exists}, Path={emb_path}")
    if exists:
        found += 1
        # Try to load
        try:
            emb = torch.load(emb_path, map_location='cpu')
            print(f"  Loaded! Shape: {emb.shape}")
        except Exception as e:
            print(f"  Error loading: {e}")

print(f"\nSummary: Found {found}/5")
