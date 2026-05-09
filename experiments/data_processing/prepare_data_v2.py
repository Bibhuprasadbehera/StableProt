import os
import torch
import pandas as pd
import hashlib
from Bio import SeqIO
from tqdm import tqdm

def get_seq_hash(seq, max_len=1500):
    # ESM-2 generation uses SHA-256 on truncated sequence
    truncated_seq = seq[:max_len]
    return hashlib.sha256(truncated_seq.encode()).hexdigest()

def load_embeddings_for_fasta(fasta_path, cache_dir="esm2_embeddings_cache"):
    """
    Reads a FASTA file and loads the corresponding embeddings from the cache.
    Returns: list of dicts with {'id': id, 'seq': seq, 'embedding': tensor}
    """
    print(f"Loading embeddings for {fasta_path}...")
    records = []
    missing = 0
    total = 0
    for record in tqdm(SeqIO.parse(fasta_path, "fasta")):
        total += 1
        seq = str(record.seq)
        seq_hash = get_seq_hash(seq)
        emb_path = os.path.join(cache_dir, f"esm2_{seq_hash}.pt")
        
        if os.path.exists(emb_path):
            try:
                emb = torch.load(emb_path, weights_only=True)
                if torch.isnan(emb).any():
                    print(f"Warning: NaN detected in embedding {emb_path}. Skipping.")
                    missing += 1
                    continue
                records.append({
                    "id": record.id,
                    "seq": seq,
                    "embedding": emb
                })
            except Exception as e:
                print(f"Error loading {emb_path}: {e}")
                missing += 1
        else:
            missing += 1
            
    print(f"Loaded {len(records)}/{total} sequences. Missing: {missing}.")
    return records

def main():
    cache_dir = "experiments/esm2_embeddings_cache"
    os.makedirs("experiments", exist_ok=True)
    output_path = "new_data/prepared_data_v2.pt"
    
    # 1. Load OGT Training Data (Cleaned after CD-HIT)
    # The IDs in OGT fasta look like: >taxid|uniprotid|ogt
    print("Processing OGT data...")
    # Update to the deduplicated fasta file
    ogt_records = load_embeddings_for_fasta("new_data/ogt_training_dedup.fasta", cache_dir)
    
    train_ogt = {
        "ids": [],
        "sequences": [],
        "labels": [],
        "embeddings": []
    }
    
    for r in ogt_records:
        parts = r["id"].split("|")
        if len(parts) >= 3:
            try:
                # header is OGT|taxid|uniprotid|ogt
                ogt_val = float(parts[-1])
                train_ogt["ids"].append(r["id"])
                train_ogt["sequences"].append(r["seq"])
                train_ogt["labels"].append(ogt_val)
                train_ogt["embeddings"].append(r["embedding"])
            except ValueError:
                continue
                
    if train_ogt["embeddings"]:
        train_ogt["embeddings"] = torch.stack(train_ogt["embeddings"])
        train_ogt["labels"] = torch.tensor(train_ogt["labels"], dtype=torch.float32)
    print(f"OGT Dataset: {len(train_ogt['ids'])} samples.")

    # 2. Load ProThermDB (Test Set)
    # The IDs look like: >uniprot|tm
    print("\nProcessing ProThermDB Test data...")
    protherm_records = load_embeddings_for_fasta("new_data/prothermdb_validation.fasta", cache_dir)
    test_tm = {
        "ids": [], "sequences": [], "labels": [], "embeddings": [], "source": []
    }
    for r in protherm_records:
        parts = r["id"].split("|")
        if len(parts) >= 2:
            try:
                tm_val = float(parts[1])
                test_tm["ids"].append(r["id"])
                test_tm["sequences"].append(r["seq"])
                test_tm["labels"].append(tm_val)
                test_tm["embeddings"].append(r["embedding"])
                test_tm["source"].append("ProThermDB")
            except ValueError:
                continue
    
    # 3. Load TemBERTure Regression (Train + Val + Test)
    print("\nProcessing TemBERTure data...")
    # TemBERTure sequences FASTA: new_data/tembert_reg_sequences.fasta (Train+Val) and tembert_test_sequences.fasta
    # Wait, earlier I saved >id|tm in analyze_tembert_data.py
    tembert_reg_records = load_embeddings_for_fasta("new_data/tembert_reg_sequences.fasta", cache_dir)
    tembert_test_records = load_embeddings_for_fasta("new_data/tembert_test_sequences.fasta", cache_dir)
    
    train_tm = {
        "ids": [], "sequences": [], "labels": [], "embeddings": [], "source": []
    }
    
    # We need to split TemBERTure into train and val using the original files
    try:
        tembert_val_df = pd.read_csv("TemBERTure_repo/data/TemBERTureVal_reg.txt")
        val_ids = set(tembert_val_df["Protein_ID"].astype(str))
    except Exception as e:
        print(f"Warning: Could not load TemBERTure validation split: {e}. Will put all in train.")
        val_ids = set()
    
    val_tm = {
        "ids": [], "sequences": [], "labels": [], "embeddings": [], "source": []
    }
    
    for r in tembert_reg_records:
        parts = r["id"].split("|")
        if len(parts) >= 2:
            uid = parts[0]
            try:
                tm_val = float(parts[1])
                if uid in val_ids:
                    target_dict = val_tm
                else:
                    target_dict = train_tm
                
                target_dict["ids"].append(r["id"])
                target_dict["sequences"].append(r["seq"])
                target_dict["labels"].append(tm_val)
                target_dict["embeddings"].append(r["embedding"])
                target_dict["source"].append("TemBERTure")
            except ValueError:
                continue
                
    for r in tembert_test_records:
        parts = r["id"].split("|")
        if len(parts) >= 2:
            try:
                tm_val = float(parts[1])
                test_tm["ids"].append(r["id"])
                test_tm["sequences"].append(r["seq"])
                test_tm["labels"].append(tm_val)
                test_tm["embeddings"].append(r["embedding"])
                test_tm["source"].append("TemBERTure")
            except ValueError:
                continue

    # 4. Load Meltome
    print("\nProcessing Meltome data...")
    meltome_records = load_embeddings_for_fasta("new_data/meltome_sequences.fasta", cache_dir)
    for r in meltome_records:
        parts = r["id"].split("|")
        if len(parts) >= 2:
            try:
                tm_val = float(parts[1])
                # We can put Meltome into train_tm
                train_tm["ids"].append(r["id"])
                train_tm["sequences"].append(r["seq"])
                train_tm["labels"].append(tm_val)
                train_tm["embeddings"].append(r["embedding"])
                train_tm["source"].append("Meltome")
            except ValueError:
                continue

    # Convert to tensors
    for d in [train_tm, val_tm, test_tm]:
        if d["embeddings"]:
            d["embeddings"] = torch.stack(d["embeddings"])
            d["labels"] = torch.tensor(d["labels"], dtype=torch.float32)

    print(f"\nFinal Dataset Sizes:")
    print(f"Train OGT: {len(train_ogt['ids'])}")
    print(f"Train Tm:  {len(train_tm['ids'])}")
    print(f"Val Tm:    {len(val_tm['ids'])}")
    print(f"Test Tm:   {len(test_tm['ids'])}")

    combined_dataset = {
        "train_ogt": train_ogt,
        "train_tm": train_tm,
        "val_tm": val_tm,
        "test_tm": test_tm
    }
    
    print(f"\nSaving to {output_path}...")
    torch.save(combined_dataset, output_path)
    print("Done!")

if __name__ == "__main__":
    main()
