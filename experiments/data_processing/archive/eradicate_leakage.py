import os
import sys
import torch
from Bio import SeqIO

def get_sequences_from_fasta(path):
    if not os.path.exists(path):
        return set()
    return {str(r.seq).upper() for r in SeqIO.parse(path, "fasta")}

def get_sequences_from_pt(path):
    if not os.path.exists(path):
        return set()
    data = torch.load(path, map_location='cpu', weights_only=False)
    if 'sequences' in data:
        return set(data['sequences'])
    return set()

def filter_fasta(input_path, output_path, forbidden_seqs, name):
    if not os.path.exists(input_path):
        return
        
    clean_records = []
    dropped = 0
    
    with open(input_path, "r") as f:
        for record in SeqIO.parse(f, "fasta"):
            seq = str(record.seq).upper()
            if seq in forbidden_seqs:
                dropped += 1
                continue
            clean_records.append(record)
            
    with open(output_path, "w") as f:
        SeqIO.write(clean_records, f, "fasta")
        
    print(f"[{name}] Eradicated {dropped} leaked sequences. Saved to {os.path.basename(output_path)}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "..", "new_data")
    exp_data_dir = os.path.join(base_dir, "data_processing")
    
    print("Loading test set holdouts (Forbidden Sequences)...")
    protherm_seqs = get_sequences_from_fasta(os.path.join(data_dir, "prothermdb_validation_clean.fasta"))
    fireprot_seqs = get_sequences_from_pt(os.path.join(exp_data_dir, "fireprot_holdout_prott5.pt"))
    
    forbidden_seqs = protherm_seqs.union(fireprot_seqs)
    print(f"Total forbidden sequences (ProThermDB + FireProt): {len(forbidden_seqs)}\n")
    
    # Process OGT Train
    in_ogt = os.path.join(data_dir, "ogt_training_clean.fasta")
    out_ogt = os.path.join(data_dir, "ogt_training_leak_free.fasta")
    filter_fasta(in_ogt, out_ogt, forbidden_seqs, "OGT Train")
    
    # Process Meltome Train
    in_meltome = os.path.join(data_dir, "meltome_sequences_clean.fasta")
    out_meltome = os.path.join(data_dir, "meltome_sequences_leak_free.fasta")
    filter_fasta(in_meltome, out_meltome, forbidden_seqs, "Meltome Train")
