import hashlib
import os
from Bio import SeqIO

def check_seq(fasta):
    rec = next(SeqIO.parse(fasta, "fasta"))
    seq = str(rec.seq)
    if len(seq) > 1500:
        seq = seq[:1500]
    h = hashlib.sha256(seq.encode()).hexdigest()
    print(f"Checking {fasta}...")
    print(f"  ID: {rec.id}")
    print(f"  Hash: {h}")
    p1 = f"esm2_embeddings_cache/esm2_{h}.pt"
    p2 = f"experiments/esm2_embeddings_cache/esm2_{h}.pt"
    print(f"  In root: {os.path.exists(p1)}")
    print(f"  In experiments: {os.path.exists(p2)}")

check_seq("new_data/meltome_sequences.fasta")
check_seq("new_data/tembert_reg_sequences.fasta")
check_seq("new_data/ogt_training_dedup.fasta")
