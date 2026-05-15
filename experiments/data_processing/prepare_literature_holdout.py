import os
import sys
import pandas as pd
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def create_template(csv_path):
    if not os.path.exists(csv_path):
        df = pd.DataFrame({
            "Sequence": [
                "MAVTAQAQAQAQ... (replace with real sequence)", 
                "MKLLILA... (replace with real sequence)"
            ],
            "Tm_(C)": [65.5, 42.0],
            "Organism": ["Thermus thermophilus", "Homo sapiens"],
            "Source_Paper_DOI": ["10.1038/s41586-...", "10.1126/science..."]
        })
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Created template at {csv_path}. Please populate this file manually.")
        sys.exit(0)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "..", "new_data")
    csv_path = os.path.join(data_dir, "literature_tm_holdout.csv")
    fasta_path = os.path.join(data_dir, "literature_tm_holdout.fasta")
    
    if not os.path.exists(csv_path):
        create_template(csv_path)
        
    print(f"Reading populated literature dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if len(df) <= 2 and "replace with real sequence" in df.iloc[0]["Sequence"]:
        print("Dataset is still the dummy template. Please add real data.")
        sys.exit(1)
        
    # Create FASTA
    records = []
    for i, row in df.iterrows():
        seq = str(row['Sequence']).upper().replace(" ", "")
        rec = SeqRecord(Seq(seq), id=f"LitSeq_{i}", description=f"Tm={row['Tm_(C)']}")
        records.append(rec)
        
    with open(fasta_path, "w") as f:
        SeqIO.write(records, f, "fasta")
        
    print(f"Generated FASTA: {fasta_path}")
    print("\nNext Steps:")
    print(f"1. Run ProtT5 embedding generator on {fasta_path}")
    print(f"2. Run ESM-2 embedding generator on {fasta_path}")
    print("3. Combine into literature_holdout.pt dictionary for model evaluation.")

if __name__ == "__main__":
    main()
