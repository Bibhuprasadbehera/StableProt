import os
import sys
import torch
import pandas as pd
from Bio import SeqIO
from collections import defaultdict

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def scan_fasta(filepath):
    if not os.path.exists(filepath):
        return {"status": "MISSING", "file": filepath}
    
    invalid_chars = set('BJOUXZ')
    stats = {"count": 0, "invalid_seqs": 0, "duplicates": 0, "lengths": []}
    seen = set()
    
    with open(filepath, "r") as f:
        for record in SeqIO.parse(f, "fasta"):
            stats["count"] += 1
            seq = str(record.seq).upper()
            stats["lengths"].append(len(seq))
            
            if any(c in invalid_chars for c in seq):
                stats["invalid_seqs"] += 1
                
            if seq in seen:
                stats["duplicates"] += 1
            seen.add(seq)
            
    stats["status"] = "OK" if stats["invalid_seqs"] == 0 and stats["duplicates"] == 0 else "WARNING"
    stats["min_len"] = min(stats["lengths"]) if stats["lengths"] else 0
    stats["max_len"] = max(stats["lengths"]) if stats["lengths"] else 0
    del stats["lengths"]
    return stats

def scan_csv(filepath, target_col, min_val=0, max_val=120):
    if not os.path.exists(filepath):
        return {"status": "MISSING", "file": filepath}
        
    df = pd.read_csv(filepath)
    stats = {"count": len(df), "missing_targets": 0, "out_of_bounds": 0}
    
    if target_col in df.columns:
        stats["missing_targets"] = int(df[target_col].isna().sum())
        stats["out_of_bounds"] = int(((df[target_col] < min_val) | (df[target_col] > max_val)).sum())
        stats["status"] = "OK" if stats["missing_targets"] == 0 and stats["out_of_bounds"] == 0 else "WARNING"
    else:
        stats["status"] = "ERROR: Target column missing"
        
    return stats

def check_leakage():
    print("Loading data for leakage check...")
    results = []
    
    # Load all sets
    try:
        prepared = torch.load(os.path.join(project_root, "new_data", "prepared_data_v2_leak_free.pt"), map_location='cpu', weights_only=False)
        train_tm_seqs = set(prepared['train_tm']['sequences']) if 'sequences' in prepared['train_tm'] else set()
    except Exception:
        train_tm_seqs = set()
        
    try:
        fireprot = torch.load(os.path.join(project_root, "experiments", "data_processing", "fireprot_holdout_prott5.pt"), map_location='cpu', weights_only=False)
        fireprot_seqs = set(fireprot['sequences'])
    except Exception:
        fireprot_seqs = set()
        
    # FASTA loads
    def get_fasta_seqs(path):
        if not os.path.exists(path): return set()
        return {str(r.seq).upper() for r in SeqIO.parse(path, "fasta")}
        
    protherm_seqs = get_fasta_seqs(os.path.join(project_root, "new_data", "prothermdb_validation_clean.fasta"))
    ogt_train_seqs = get_fasta_seqs(os.path.join(project_root, "new_data", "ogt_training_leak_free.fasta"))
    
    # 1. FireProt vs Train Tm
    if train_tm_seqs and fireprot_seqs:
        overlap = train_tm_seqs.intersection(fireprot_seqs)
        results.append(f"- **Train Tm vs FireProt OOD**: {len(overlap)} exact overlaps detected.")
        if overlap: results.append(f"  - *CRITICAL LEAKAGE*: {list(overlap)[:3]}...")
        
    # 2. ProThermDB vs OGT Train
    if ogt_train_seqs and protherm_seqs:
        overlap = ogt_train_seqs.intersection(protherm_seqs)
        results.append(f"- **OGT Train vs ProThermDB Val**: {len(overlap)} exact overlaps detected.")
        if overlap: results.append(f"  - *CRITICAL LEAKAGE*: Overlap violates strict generalization bounds.")
        
    return results

def main():
    report_path = os.path.join(project_root, "testing_results.md")
    
    print("Running Exhaustive Data Validation...")
    
    with open(report_path, "w") as f:
        f.write("# StableProt V2: Exhaustive Data Validation Results\n\n")
        f.write("> **Note**: This report tracks structural, biological, and leakage validations for all core datasets.\n\n")
        
        # 1. FASTA Validations
        f.write("## 1. Sequence Validations (FASTA)\n\n")
        fastas = [
            ("new_data/ogt_training_leak_free.fasta", "OGT Train (Cleaned)"),
            ("new_data/prothermdb_validation_clean.fasta", "ProThermDB Val (Cleaned)"),
            ("new_data/meltome_sequences_leak_free.fasta", "Meltome Train (Cleaned)")
        ]
        
        for path, name in fastas:
            full_path = os.path.join(project_root, path)
            res = scan_fasta(full_path)
            f.write(f"### {name}\n")
            if res["status"] == "MISSING":
                f.write(f"- ❌ File missing: `{path}`\n\n")
            else:
                f.write(f"- Total Sequences: {res['count']}\n")
                f.write(f"- Min/Max Length: {res['min_len']} - {res['max_len']}\n")
                if res["invalid_seqs"] > 0:
                    f.write(f"- ❌ **Invalid Amino Acids**: {res['invalid_seqs']} sequences contain B, J, O, U, X, or Z.\n")
                if res["duplicates"] > 0:
                    f.write(f"- ❌ **Exact Duplicates**: {res['duplicates']} sequences are duplicated in this file.\n")
                if res["status"] == "OK":
                    f.write("- ✅ Sequence Integrity: PASS\n")
                f.write("\n")
                
        # 2. Label Validations
        f.write("## 2. Label Validations (CSV)\n\n")
        csvs = [
            ("new_data/meltome_sequences_with_ogt.csv", "OGT", 0, 110),
            ("new_data/prothermdb_validation_with_ogt.csv", "Tm_(C)", 0, 110)
        ]
        
        for path, target_col, min_v, max_v in csvs:
            full_path = os.path.join(project_root, path)
            res = scan_csv(full_path, target_col, min_v, max_v)
            f.write(f"### {path}\n")
            if res["status"] == "MISSING":
                f.write(f"- ❌ File missing\n\n")
            elif "ERROR" in res["status"]:
                f.write(f"- ❌ {res['status']}\n\n")
            else:
                f.write(f"- Total Rows: {res['count']}\n")
                if res["missing_targets"] > 0:
                    f.write(f"- ❌ **Missing Labels (NaN)**: {res['missing_targets']} rows\n")
                if res["out_of_bounds"] > 0:
                    f.write(f"- ❌ **Out of Bounds (<{min_v} or >{max_v})**: {res['out_of_bounds']} rows\n")
                if res["status"] == "OK":
                    f.write("- ✅ Label Integrity: PASS\n")
                f.write("\n")
                
        # 3. Leakage Checks
        f.write("## 3. Cross-Dataset Leakage (Exact Match)\n\n")
        leakage_results = check_leakage()
        for r in leakage_results:
            f.write(r + "\n")
            
    print(f"Validation complete. Results saved to {report_path}")

if __name__ == "__main__":
    main()
