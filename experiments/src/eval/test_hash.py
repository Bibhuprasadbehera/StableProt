import hashlib
from Bio import SeqIO
import os

def get_seq_hash(seq):
    return hashlib.sha256(seq.encode()).hexdigest()

def read_fasta_custom(file_path):
    sequences = []
    with open(file_path, 'r') as f:
        seq_id = ""
        seq = ""
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if seq_id: sequences.append((seq_id, seq))
                seq_id = line[1:]; seq = ""
            else: seq += line
        if seq_id: sequences.append((seq_id, seq))
    return sequences

fasta_path = "new_data/ogt_training_dedup.fasta"
print("Reading with SeqIO...")
for record in SeqIO.parse(fasta_path, "fasta"):
    seq_io = str(record.seq)
    hash_io = get_seq_hash(seq_io)
    print(f"SeqIO Hash: {hash_io}")
    break

print("\nReading with Custom...")
custom_seqs = read_fasta_custom(fasta_path)
sid, seq_custom = custom_seqs[0]
hash_custom = get_seq_hash(seq_custom)
print(f"Custom Hash: {hash_custom}")

cache_dir = "esm2_embeddings_cache"
path_io = os.path.join(cache_dir, f"esm2_{hash_io}.pt")
path_custom = os.path.join(cache_dir, f"esm2_{hash_custom}.pt")

print(f"\nSeqIO Path exists: {os.path.exists(path_io)}")
print(f"Custom Path exists: {os.path.exists(path_custom)}")
