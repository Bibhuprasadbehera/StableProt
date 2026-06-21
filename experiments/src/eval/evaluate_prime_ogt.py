#!/usr/bin/env python3
"""
Evaluate the PRIME (AI4Protein/Prime_690M) model on the OGT test split.
"""

import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_FILE = PROJECT_ROOT / "data" / "embeddings" / "prepared_data_v7_saprot1.3b_seqonly_ogt_split.pt"
OUTPUT_FILE = PROJECT_ROOT / "experiments" / "src" / "eval" / "ogt_baselines" / "prime_predictions.pt"

def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    pcc, _ = pearsonr(y_true, y_pred)
    spearman, _ = spearmanr(y_true, y_pred)
    
    print(f"--- {name} ---")
    print(f"  MAE:      {mae:.4f}")
    print(f"  RMSE:     {rmse:.4f}")
    print(f"  PCC:      {pcc:.4f}")
    print(f"  Spearman: {spearman:.4f}\n")
    return {'mae': mae, 'rmse': rmse, 'pcc': pcc, 'spearman': spearman}

def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    print("Loading OGT test split...")
    data = torch.load(DATA_FILE, map_location='cpu', weights_only=False)
    
    test_sequences = data['test_ogt']['sequences']
    test_labels = data['test_ogt']['ogt_consensus']
    if isinstance(test_labels, torch.Tensor):
        test_labels = test_labels.numpy()
        
    print(f"Total test sequences: {len(test_sequences)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading PRIME model on {device}...")
    model_path = "AI4Protein/Prime_690M"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        model.eval()
        model.to(device)
    except Exception as e:
        print(f"Failed to load PRIME model: {e}")
        return
        
    print("Running inference...")
    batch_size = 16
    y_pred = []
    
    t0 = time.time()
    with torch.no_grad():
        for i in tqdm(range(0, len(test_sequences), batch_size)):
            batch_seqs = test_sequences[i:i+batch_size]
            inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # PRIME outputs .predicted_values for OGT
            logits = model(**inputs).predicted_values
            
            # Convert to list
            y_pred.extend(logits.squeeze(-1).cpu().numpy().tolist())
            
    print(f"Inference completed in {time.time()-t0:.1f}s")
    
    y_pred = np.array(y_pred)
    
    results = evaluate(test_labels, y_pred, "PRIME (690M)")
    
    # Save predictions
    if OUTPUT_FILE.exists():
        existing = torch.load(OUTPUT_FILE, map_location='cpu')
    else:
        existing = {'y_true': test_labels}
        
    existing['PRIME'] = y_pred
    torch.save(existing, OUTPUT_FILE)
    print(f"Saved PRIME predictions to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
