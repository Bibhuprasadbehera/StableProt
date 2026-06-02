import os
import torch
import numpy as np
import json
from model import StableProtV7
from config import CONFIG

def load_compat_state_dict(model, state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("backbone.0."):
            new_state_dict[k.replace("backbone.0.", "input_layer.")] = v
        elif k.startswith("backbone.1."):
            new_state_dict[k.replace("backbone.1.", "backbone_rest.0.")] = v
        elif k.startswith("backbone.2."):
            new_state_dict[k.replace("backbone.2.", "backbone_rest.1.")] = v
        elif k.startswith("backbone.3."):
            new_state_dict[k.replace("backbone.3.", "backbone_rest.2.")] = v
        elif k.startswith("backbone.4."):
            new_state_dict[k.replace("backbone.4.", "backbone_rest.3.")] = v
        elif k.startswith("backbone.5."):
            new_state_dict[k.replace("backbone.5.", "backbone_rest.4.")] = v
        elif k.startswith("backbone.6."):
            new_state_dict[k.replace("backbone.6.", "backbone_rest.5.")] = v
        elif k.startswith("backbone.7."):
            new_state_dict[k.replace("backbone.7.", "backbone_rest.6.")] = v
        elif k.startswith("backbone.8."):
            new_state_dict[k.replace("backbone.8.", "backbone_rest.7.")] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)

def main():
    base_dir = "/home/bibhu/Documents/temstampto"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load dataset (ESM-2 manually cleaned raw version)
    data_path = os.path.join(base_dir, "data/embeddings/prepared_data_ogt_manual_clean_temp.pt")
    print(f"Loading ESM-2 temp dataset from {data_path}...")
    dataset = torch.load(data_path, map_location='cpu')
    
    train_ogt = dataset["train_ogt"]
    embeddings = train_ogt["embeddings"]
    labels = train_ogt["labels"] if "labels" in train_ogt else train_ogt["ogt_original"]
    ids = train_ogt["ids"]
    tax_ids = train_ogt["tax_id"]
    
    # Load Stage 1 model (seed 1) to get predictions
    model_path = os.path.join(base_dir, "experiments/src/training/v7_transfer/results/seed1/model_stage1.pt")
    print(f"Loading Stage 1 model from {model_path}...")
    model = StableProtV7(
        emb_dim=2560,
        hidden=CONFIG['hidden_size'],
        bottleneck=CONFIG['bottleneck_size']
    ).to(device)
    
    sd = torch.load(model_path, map_location=device)
    load_compat_state_dict(model, sd)
    model.eval()
    
    # Run prediction in batches
    batch_size = 4096
    total = len(embeddings)
    preds = []
    print(f"Running predictions on {total} OGT sequences...")
    with torch.no_grad():
        for i in range(0, total, batch_size):
            batch_emb = embeddings[i:i+batch_size].to(device)
            p = model(batch_emb, stage='ogt').cpu().tolist()
            preds.extend(p)
            
    preds = np.array(preds)
    labels = np.array(labels)
    errors = np.abs(preds - labels)
    
    # 2. Filter clean indices (error <= 15°C)
    clean_indices = np.where(errors <= 15.0)[0]
    discard_indices = np.where(errors > 15.0)[0]
    print(f"Clean sequences (error <= 15°C): {len(clean_indices)} ({len(clean_indices)/total*100:.2f}%)")
    print(f"Discarded sequences (error > 15°C): {len(discard_indices)} ({len(discard_indices)/total*100:.2f}%)")
    
    # Track which TaxIDs are discarded/clean
    clean_tax_ids = set()
    discarded_tax_ids = set()
    for idx in clean_indices:
        clean_tax_ids.add(tax_ids[idx])
    for idx in discard_indices:
        discarded_tax_ids.add(tax_ids[idx])
        
    # Strictly discarded organisms are those that have discarded sequences and NO clean sequences
    strictly_discarded_tax_ids = discarded_tax_ids - clean_tax_ids
    print(f"Clean TaxIDs: {len(clean_tax_ids)}, Strictly Discarded TaxIDs: {len(strictly_discarded_tax_ids)}")
    
    # Map from taxid to OGT predicted
    taxid_to_pred_ogt = {}
    for i, idx in enumerate(clean_indices):
        taxid = tax_ids[idx]
        if taxid not in taxid_to_pred_ogt:
            taxid_to_pred_ogt[taxid] = []
        taxid_to_pred_ogt[taxid].append(preds[idx])
    taxid_to_pred_ogt = {k: float(np.mean(v)) for k, v in taxid_to_pred_ogt.items()}
    
    # 3. Split clean OGT into train/val/test (98/1/1)
    np.random.seed(42)
    shuffled_indices = clean_indices.copy()
    np.random.shuffle(shuffled_indices)
    
    n_clean = len(shuffled_indices)
    n_val = int(0.01 * n_clean)
    n_test = int(0.01 * n_clean)
    n_train = n_clean - n_val - n_test
    
    train_idx = shuffled_indices[:n_train]
    val_idx = shuffled_indices[n_train:n_train+n_val]
    test_idx = shuffled_indices[n_train+n_val:]
    
    print(f"Splits: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    def extract_split(indices):
        split = {}
        for k, v in train_ogt.items():
            if isinstance(v, torch.Tensor):
                split[k] = v[indices]
            elif isinstance(v, list):
                split[k] = [v[idx] for idx in indices]
            else:
                split[k] = v
        return split
        
    cleaned_dataset = {
        "train_ogt": extract_split(train_idx),
        "val_ogt": extract_split(val_idx),
        "test_ogt": extract_split(test_idx)
    }
    
    # Load SaProt dataset (manually cleaned raw version)
    saprot_data_path = os.path.join(base_dir, "data/embeddings/prepared_data_ogt_manual_clean_temp_saprot.pt")
    print(f"Loading SaProt temp dataset from {saprot_data_path}...")
    saprot_dataset = torch.load(saprot_data_path, map_location='cpu')
    saprot_train_ogt = saprot_dataset["train_ogt"]
    
    def extract_saprot_split(indices):
        split = {}
        for k, v in saprot_train_ogt.items():
            if isinstance(v, torch.Tensor):
                split[k] = v[indices]
            elif isinstance(v, list):
                split[k] = [v[idx] for idx in indices]
            else:
                split[k] = v
        return split
        
    cleaned_saprot_dataset = {
        "train_ogt": extract_saprot_split(train_idx),
        "val_ogt": extract_saprot_split(val_idx),
        "test_ogt": extract_saprot_split(test_idx)
    }
    
    # 4. Process TM splits for both datasets
    with open(os.path.join(base_dir, "data/cleaner_data/tm_ogt_lookup.json"), "r") as f:
        tm_ogt_lookup = json.load(f)
        
    for split_name in ["train_tm", "val_tm", "test_tm"]:
        print(f"Processing TM split: {split_name}...")
        
        # ESM-2 TM split
        tm_split = dataset[split_name]
        # SaProt TM split
        saprot_tm_split = saprot_dataset[split_name]
        
        tm_ids = tm_split["ids"]
        new_ogt_features = []
        
        for i, full_id in enumerate(tm_ids):
            uniprot_id = full_id.split("|")[0]
            lookup_info = tm_ogt_lookup.get(uniprot_id, {})
            taxid = lookup_info.get("taxid", "")
            
            if taxid in strictly_discarded_tax_ids or taxid not in clean_tax_ids:
                pred_ogt = taxid_to_pred_ogt.get(taxid, 37.0)
                new_ogt_features.append(pred_ogt)
            else:
                db_ogt = tm_split["ogt"][i].item() if "ogt" in tm_split else float(lookup_info.get("ogt", 37.0))
                new_ogt_features.append(db_ogt)
                
        tm_split["ogt"] = torch.tensor(new_ogt_features, dtype=torch.float32)
        saprot_tm_split["ogt"] = torch.tensor(new_ogt_features, dtype=torch.float32)
        
        cleaned_dataset[split_name] = tm_split
        cleaned_saprot_dataset[split_name] = saprot_tm_split
        
    # Save final cleaned datasets
    esm2_out_path = os.path.join(base_dir, "data/embeddings/prepared_data_ogt_manual_clean.pt")
    saprot_out_path = os.path.join(base_dir, "data/embeddings/prepared_data_ogt_manual_clean_saprot.pt")
    
    print(f"Saving final cleaned ESM-2 dataset to {esm2_out_path}...")
    torch.save(cleaned_dataset, esm2_out_path)
    
    print(f"Saving final cleaned SaProt dataset to {saprot_out_path}...")
    torch.save(cleaned_saprot_dataset, saprot_out_path)
    
    print("All datasets manually cleaned and filtered successfully!")
    
    # Clean up temp files
    try:
        os.remove(data_path)
        os.remove(saprot_data_path)
        print("Removed temporary files.")
    except Exception as e:
        print(f"Error removing temp files: {e}")

if __name__ == "__main__":
    main()
