import os
import hashlib
from Bio import SeqIO

def get_seq_hash(seq, max_len=1500):
    truncated_seq = seq[:max_len]
    return hashlib.sha256(truncated_seq.encode()).hexdigest()

cache_dir = "experiments/esm2_embeddings_cache"

fastas = [
    "new_data/meltome_sequences.fasta",
    "new_data/tembert_reg_sequences.fasta",
    "new_data/tembert_test_sequences.fasta",
    "new_data/prothermdb_validation.fasta"
]

for fasta in fastas:
    if not os.path.exists(fasta):
        print(f"File {fasta} not found.")
        continue
    found = 0
    total = 0
    for record in SeqIO.parse(fasta, "fasta"):
        total += 1
        seq = str(record.seq)
        seq_hash = get_seq_hash(seq)
        emb_path = os.path.join(cache_dir, f"esm2_{seq_hash}.pt")
        if os.path.exists(emb_path):
            found += 1
    print(f"{fasta}: Found {found}/{total}")
