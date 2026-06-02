import os
import torch
import numpy as np
import json

# Kyte-Doolittle Hydropathy Scale
KD_SCALE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

def is_disordered_uversky(seq):
    seq = seq.upper()
    n = len(seq)
    if n == 0:
        return False
    
    # 1. Mean Hydropathy
    h_sum = sum(KD_SCALE.get(aa, 0.0) for aa in seq)
    mean_h = h_sum / n
    # Normalize KD to [0, 1] range: KD_norm = (KD + 4.5) / 9.0
    mean_h_norm = (mean_h + 4.5) / 9.0
    
    # 2. Mean Net Charge
    pos = seq.count('R') + seq.count('K')
    neg = seq.count('D') + seq.count('E')
    mean_r = abs(pos - neg) / n
    
    # 3. Uversky Discriminant
    boundary = (mean_r + 1.151) / 2.785
    return mean_h_norm < boundary

def clean_dataset(data_path, out_path, groel_clients):
    print(f"\nLoading dataset from {data_path}...")
    dataset = torch.load(data_path, map_location='cpu')
    
    train_ogt = dataset["train_ogt"]
    embeddings = train_ogt["embeddings"]
    labels = train_ogt["labels"] if "labels" in train_ogt else train_ogt["ogt_original"]
    sequences = train_ogt["sequences"]
    ids = train_ogt["ids"]
    tax_ids = train_ogt["tax_id"]
    
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    labels = np.array(labels)
    
    total = len(ids)
    print(f"Total OGT sequences before cleaning: {total}")
    
    keep_indices = []
    stats = {
        "cys_outliers": 0,
        "asn_outliers": 0,
        "gln_outliers": 0,
        "length_outliers": 0,
        "disorder_outliers": 0,
        "groel_outliers": 0,
    }
    
    for i in range(total):
        seq = sequences[i]
        temp = labels[i]
        full_id = ids[i]
        
        # Extract UniProt ID (usually second part in taxid|uniprot format)
        parts = full_id.split("|")
        uniprot_id = parts[1] if len(parts) > 1 else parts[0]
        
        # 1. GroEL obligate client filter
        if uniprot_id in groel_clients:
            stats["groel_outliers"] += 1
            continue
            
        # 2. Length limits
        seq_len = len(seq)
        if temp >= 80.0 and seq_len > 800:
            stats["length_outliers"] += 1
            continue
        elif temp >= 60.0 and seq_len > 1000:
            stats["length_outliers"] += 1
            continue
            
        # 3. Thermolabile residue composition
        cys_pct = seq.count('C') / seq_len * 100
        asn_pct = seq.count('N') / seq_len * 100
        gln_pct = seq.count('Q') / seq_len * 100
        
        if temp >= 80.0:
            if cys_pct > 3.0:
                stats["cys_outliers"] += 1
                continue
            if asn_pct > 7.0:
                stats["asn_outliers"] += 1
                continue
            if gln_pct > 5.0:
                stats["gln_outliers"] += 1
                continue
        elif temp >= 60.0:
            if cys_pct > 4.0:
                stats["cys_outliers"] += 1
                continue
            if asn_pct > 8.0:
                stats["asn_outliers"] += 1
                continue
            if gln_pct > 6.0:
                stats["gln_outliers"] += 1
                continue
                
        # 4. Intrinsically Disordered Protein (Uversky Discriminant)
        if is_disordered_uversky(seq):
            stats["disorder_outliers"] += 1
            continue
            
        # If passed all checks, keep the sequence
        keep_indices.append(i)
        
    keep_indices = np.array(keep_indices)
    kept_count = len(keep_indices)
    
    print("\n--- Outlier Statistics ---")
    print(f"GroEL obligate clients removed: {stats['groel_outliers']}")
    print(f"Sequence length outliers:       {stats['length_outliers']}")
    print(f"Cys composition outliers:       {stats['cys_outliers']}")
    print(f"Asn composition outliers:       {stats['asn_outliers']}")
    print(f"Gln composition outliers:       {stats['gln_outliers']}")
    print(f"Intrinsically disordered (IDPs): {stats['disorder_outliers']}")
    print(f"Total discarded:                {total - kept_count} ({(total - kept_count)/total*100:.2f}%)")
    print(f"Total kept:                     {kept_count} ({kept_count/total*100:.2f}%)")
    
    # Slice the dataset dictionaries using keep_indices
    cleaned_train_ogt = {}
    for k, v in train_ogt.items():
        if isinstance(v, torch.Tensor):
            cleaned_train_ogt[k] = v[keep_indices]
        elif isinstance(v, list):
            cleaned_train_ogt[k] = [v[idx] for idx in keep_indices]
        else:
            cleaned_train_ogt[k] = v
            
    # Assemble final dataset dict (preserve validation/test and other splits as is)
    cleaned_dataset = {}
    for k, v in dataset.items():
        if k == "train_ogt":
            cleaned_dataset[k] = cleaned_train_ogt
        else:
            cleaned_dataset[k] = v
            
    print(f"Saving cleaned dataset to {out_path}...")
    torch.save(cleaned_dataset, out_path)
    print("Done!")
    return kept_count

def main():
    base_dir = "/home/bibhu/Documents/temstampto"
    
    # Load GroEL clients
    groel_path = os.path.join(base_dir, "data/training_data/ogt/groel_obligate_clients.json")
    if os.path.exists(groel_path):
        with open(groel_path, "r") as f:
            groel_clients = set(json.load(f))
    else:
        print(f"WARNING: GroEL obligate client list missing at {groel_path}")
        groel_clients = set()
        
    # Process ESM-2
    esm2_in = os.path.join(base_dir, "data/embeddings/prepared_data_v4.pt")
    esm2_out = os.path.join(base_dir, "data/embeddings/prepared_data_ogt_manual_clean.pt")
    if os.path.exists(esm2_in):
        clean_dataset(esm2_in, esm2_out, groel_clients)
        
    # Process SaProt
    saprot_in = os.path.join(base_dir, "data/embeddings/prepared_data_v4_saprot.pt")
    saprot_out = os.path.join(base_dir, "data/embeddings/prepared_data_ogt_manual_clean_saprot.pt")
    if os.path.exists(saprot_in):
        clean_dataset(saprot_in, saprot_out, groel_clients)

if __name__ == "__main__":
    main()
