import os
from Bio import SeqIO

def sanitize_fasta(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return
        
    invalid_chars = set('BJOUXZ')
    clean_records = []
    seen_seqs = set()
    
    stats = {"original": 0, "invalid_char": 0, "length_outlier": 0, "duplicate": 0, "clean": 0}
    
    with open(input_path, "r") as f:
        for record in SeqIO.parse(f, "fasta"):
            stats["original"] += 1
            seq = str(record.seq).upper()
            
            # 1. Invalid Characters
            if any(c in invalid_chars for c in seq):
                stats["invalid_char"] += 1
                continue
                
            # 2. Length Constraints
            seq_len = len(seq)
            if seq_len < 20 or seq_len > 1500:
                stats["length_outlier"] += 1
                continue
                
            # 3. Exact Duplicates
            if seq in seen_seqs:
                stats["duplicate"] += 1
                continue
                
            seen_seqs.add(seq)
            clean_records.append(record)
            stats["clean"] += 1
            
    with open(output_path, "w") as f:
        SeqIO.write(clean_records, f, "fasta")
        
    print(f"Sanitized {os.path.basename(input_path)}:")
    print(f"  Original: {stats['original']}")
    print(f"  Dropped (Invalid Char): {stats['invalid_char']}")
    print(f"  Dropped (Length Outlier): {stats['length_outlier']}")
    print(f"  Dropped (Duplicate): {stats['duplicate']}")
    print(f"  Final Clean: {stats['clean']} -> Saved to {os.path.basename(output_path)}\n")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "..", "new_data")
    
    targets = [
        ("ogt_training_dedup.fasta", "ogt_training_clean.fasta"),
        ("prothermdb_validation.fasta", "prothermdb_validation_clean.fasta"),
        ("meltome_sequences.fasta", "meltome_sequences_clean.fasta")
    ]
    
    for in_file, out_file in targets:
        sanitize_fasta(os.path.join(data_dir, in_file), os.path.join(data_dir, out_file))
