import os
import subprocess
from Bio import SeqIO

def run_cdhit(input_fasta, output_fasta, identity=0.4):
    print(f"Running CD-HIT with {identity*100}% identity threshold...")
    cmd = [
        "cd-hit",
        "-i", input_fasta,
        "-o", output_fasta,
        "-c", str(identity),
        "-n", "2",
        "-M", "16000", # 16GB memory
        "-T", "0"      # All threads
    ]
    subprocess.run(cmd, check=True)
    print(f"CD-HIT completed. Output saved to {output_fasta}")

def read_fasta_ids(fasta_path):
    ids = set()
    for record in SeqIO.parse(fasta_path, "fasta"):
        ids.add(record.id)
    return ids

def main():
    # 1. Combine OGT training + ProThermDB test + Tm validation
    # Wait, we need to know which ones are ProThermDB and which are OGT
    # We can prefix the IDs
    
    combined_fasta = "new_data/cdhit_combined_input.fasta"
    print(f"Creating combined FASTA: {combined_fasta}")
    
    with open(combined_fasta, "w") as out:
        # OGT Training
        print("Adding OGT training...")
        ogt_path = "dataset/TemStaPro-Major-30-imbal-training.fasta"
        for record in SeqIO.parse(ogt_path, "fasta"):
            out.write(f">OGT|{record.id}\n{str(record.seq)}\n")
            
        # ProThermDB Validation (which is our test set)
        print("Adding ProThermDB test...")
        protherm_path = "new_data/prothermdb_validation.fasta"
        for record in SeqIO.parse(protherm_path, "fasta"):
            out.write(f">PROTHERM|{record.id}\n{str(record.seq)}\n")
            
        # TemBERTure Regression (Tm validation/training)
        print("Adding TemBERTure Tm...")
        tembert_path = "new_data/tembert_reg_sequences.fasta"
        for record in SeqIO.parse(tembert_path, "fasta"):
            out.write(f">TEMBERT|{record.id}\n{str(record.seq)}\n")
            
        # Meltome
        print("Adding Meltome Tm...")
        meltome_path = "new_data/meltome_sequences.fasta"
        for record in SeqIO.parse(meltome_path, "fasta"):
            out.write(f">MELTOME|{record.id}\n{str(record.seq)}\n")
            
    # 2. Run CD-HIT
    cdhit_out = "new_data/cdhit_combined_output.fasta"
    run_cdhit(combined_fasta, cdhit_out, identity=0.4)
    
    # 3. Parse clusters to find OGT sequences to remove
    clstr_file = f"{cdhit_out}.clstr"
    print(f"Parsing CD-HIT clusters from {clstr_file}...")
    
    ogt_to_remove = set()
    current_cluster = []
    
    with open(clstr_file, "r") as f:
        for line in f:
            if line.startswith(">"):
                # Process previous cluster
                has_test_or_val = any(id.startswith("PROTHERM|") or id.startswith("TEMBERT|") or id.startswith("MELTOME|") for id in current_cluster)
                if has_test_or_val:
                    for id in current_cluster:
                        if id.startswith("OGT|"):
                            # Remove the OGT| prefix when saving
                            ogt_to_remove.add(id[4:])
                current_cluster = []
            else:
                # Example line: 0	449aa, >OGT|tr|A0A024RBG1|A0A024RBG1_HUMAN|37.0... at 100%
                parts = line.split(">")
                if len(parts) > 1:
                    seq_id = parts[1].split("...")[0]
                    current_cluster.append(seq_id)
                    
    # Process last cluster
    has_test_or_val = any(id.startswith("PROTHERM|") or id.startswith("TEMBERT|") or id.startswith("MELTOME|") for id in current_cluster)
    if has_test_or_val:
        for id in current_cluster:
            if id.startswith("OGT|"):
                ogt_to_remove.add(id[4:])
                
    print(f"Found {len(ogt_to_remove)} OGT training sequences to remove due to >= 40% identity with test/val sets.")
    
    # 4. Create filtered OGT training set
    filtered_ogt = "new_data/ogt_training_dedup.fasta"
    print(f"Creating filtered OGT training set: {filtered_ogt}")
    
    removed_count = 0
    kept_count = 0
    
    with open(filtered_ogt, "w") as out:
        for record in SeqIO.parse(ogt_path, "fasta"):
            if record.id in ogt_to_remove:
                removed_count += 1
            else:
                out.write(f">{record.id}\n{str(record.seq)}\n")
                kept_count += 1
                
    print(f"Done. Kept: {kept_count}, Removed: {removed_count}")
    
if __name__ == "__main__":
    main()
