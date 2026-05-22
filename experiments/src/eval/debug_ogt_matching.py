import os
import hashlib
from Bio import SeqIO

def get_seq_hash(seq, max_len=1500):
    truncated_seq = seq[:max_len]
    return hashlib.sha256(truncated_seq.encode()).hexdigest()

cache_dir = "experiments/esm2_embeddings_cache"
ogt_fasta = "new_data/ogt_training_dedup.fasta"

print(f"Checking first 5 sequences from {ogt_fasta}")
for i, record in enumerate(SeqIO.parse(ogt_fasta, "fasta")):
    if i >= 5: break
    seq = str(record.seq)
    seq_hash = get_seq_hash(seq)
    emb_path = os.path.join(cache_dir, f"esm2_{seq_hash}.pt")
    exists = os.path.exists(emb_path)
    print(f"Seq {i}: Hash={seq_hash}, Exists={exists}")
