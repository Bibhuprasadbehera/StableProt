import torch
import numpy as np
from collections import defaultdict
import os

def main():
    input_path = "/home/bibhu/Documents/temstampto/new_data/prepared_data_v2.pt"
    output_path = "/home/bibhu/Documents/temstampto/new_data/prepared_data_v3.pt"
    
    print(f"Loading dataset from {input_path}...")
    dataset = torch.load(input_path, map_location="cpu")
    
    train_tm = dataset["train_tm"]
    
    ids = train_tm["ids"]
    labels = train_tm["labels"].numpy()
    embeddings = train_tm["embeddings"]
    sequences = train_tm["sequences"]
    sources = train_tm["source"]
    
    # Group indices by base UniProt ID
    uid_to_indices = defaultdict(list)
    for i, full_id in enumerate(ids):
        base_uid = full_id.split("|")[0]
        uid_to_indices[base_uid].append(i)
        
    print(f"Original samples in train_tm: {len(ids)}")
    print(f"Unique protein IDs: {len(uid_to_indices)}")
    
    new_ids = []
    new_sequences = []
    new_labels = []
    new_embeddings = []
    new_sources = []
    
    removed_outliers = 0
    
    for base_uid, idxs in uid_to_indices.items():
        tm_vals = labels[idxs]
        
        # Check range (max - min)
        if len(tm_vals) > 1:
            val_range = np.max(tm_vals) - np.min(tm_vals)
            if val_range > 10.0:
                removed_outliers += 1
                continue
                
        # Calculate median Tm
        median_tm = np.median(tm_vals)
        
        # Use first index for embedding and sequence
        first_idx = idxs[0]
        
        # Format the new ID as base_uid|median_tm
        new_ids.append(f"{base_uid}|{median_tm:.3f}")
        new_sequences.append(sequences[first_idx])
        new_labels.append(median_tm)
        new_embeddings.append(embeddings[first_idx])
        
        # Combine sources
        unique_srcs = sorted(list(set(sources[idx] for idx in idxs)))
        new_sources.append(",".join(unique_srcs))
        
    print(f"Removed {removed_outliers} high-disagreement outliers.")
    print(f"New samples in train_tm: {len(new_ids)}")
    
    new_train_tm = {
        "ids": new_ids,
        "sequences": new_sequences,
        "labels": torch.tensor(new_labels, dtype=torch.float32),
        "embeddings": torch.stack(new_embeddings),
        "source": new_sources
    }
    
    # Save the new dataset
    dataset["train_tm"] = new_train_tm
    
    print(f"Saving to {output_path}...")
    torch.save(dataset, output_path)
    print("Successfully curated prepared_data_v3.pt!")

if __name__ == "__main__":
    main()
