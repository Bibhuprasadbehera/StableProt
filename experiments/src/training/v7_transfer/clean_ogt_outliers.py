import os
import torch
import numpy as np
from model import StableProtV7
from config import CONFIG

def main():
    base_dir = "../../../"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    data_path = "/home/bibhu/Documents/temstampto/data/embeddings/prepared_data_v4.pt"
    print(f"Loading dataset from {data_path}...")
    dataset = torch.load(data_path, map_location='cpu')
    
    train_ogt = dataset["train_ogt"]
    embeddings = train_ogt["embeddings"]  # 2560-dim
    
    # Handle labels
    if "labels" in train_ogt:
        labels = train_ogt["labels"]
    elif "ogt_original" in train_ogt:
        labels = train_ogt["ogt_original"]
    else:
        labels = dataset["train_ogt"]["labels"]
        
    ids = train_ogt["ids"]
    
    # Load Stage 1 model (seed 1)
    model_path = "results/seed1/model_stage1.pt"
    print(f"Loading Stage 1 model from {model_path}...")
    model = StableProtV7(
        emb_dim=2560,
        hidden=CONFIG['hidden_size'],
        bottleneck=CONFIG['bottleneck_size']
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
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
            if i % (batch_size * 20) == 0:
                print(f"  Processed {i}/{total}...")
                
    preds = np.array(preds)
    labels = np.array(labels)
    errors = np.abs(preds - labels)
    
    print("\n--- Error Distribution Summary ---")
    print(f"Mean Absolute Error: {np.mean(errors):.4f}°C")
    print(f"Median Absolute Error: {np.median(errors):.4f}°C")
    
    thresholds = [5, 10, 15, 20]
    for t in thresholds:
        count = np.sum(errors > t)
        pct = (count / total) * 100
        print(f"Error > {t}°C: {count} sequences ({pct:.2f}%)")
        
    # Find the top 5 organisms (by taxid) with the highest average error
    taxid_to_errors = {}
    for i, full_id in enumerate(ids):
        taxid = full_id.split("|")[0]
        if taxid not in taxid_to_errors:
            taxid_to_errors[taxid] = []
        taxid_to_errors[taxid].append(errors[i])
        
    taxid_mean_errors = []
    for taxid, errs in taxid_to_errors.items():
        if len(errs) >= 50:  # only look at organisms with at least 50 proteins
            taxid_mean_errors.append((taxid, np.mean(errs), len(errs)))
            
    taxid_mean_errors.sort(key=lambda x: x[1], reverse=True)
    
    print("\n--- Top 10 Organisms with Highest Average Error (min 50 proteins) ---")
    for taxid, mean_err, count in taxid_mean_errors[:10]:
        print(f"TaxID: {taxid:<10} | Mean Error: {mean_err:.2f}°C | Count: {count}")

if __name__ == "__main__":
    main()
