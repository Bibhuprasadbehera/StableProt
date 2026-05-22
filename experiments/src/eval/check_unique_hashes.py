import hashlib
from Bio import SeqIO
from tqdm import tqdm

fasta_path = "new_data/ogt_training_dedup.fasta"
hashes = set()
count = 0
for record in tqdm(SeqIO.parse(fasta_path, "fasta")):
    seq = str(record.seq)
    seq_hash = hashlib.sha256(seq.encode()).hexdigest()
    hashes.add(seq_hash)
    count += 1

print(f"Total sequences: {count}")
print(f"Unique hashes: {len(hashes)}")
