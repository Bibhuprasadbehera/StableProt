import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error

sys.path.append("experiments/src/training/v9_disjoint")
from train import MultiHeadSaProtV8, enrich_inputs, sanitize_data, ProteinDataset, MesophilicSubsampler, compute_bin_weights

def evaluate_val_ogt():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir = "experiments/src/training/v9_disjoint/results"
    
    norm_stats = torch.load(os.path.join(results_dir, "normalization_stats.pt"), map_location="cpu", weights_only=False)
    ogt_mean = norm_stats['ogt_mean']
    ogt_std = norm_stats['ogt_std']

    print("Loading data/embeddings/prepared_data_v3.pt and val_ogt split...")
    data = torch.load("data/embeddings/prepared_data_v3.pt", map_location="cpu", weights_only=False)
    ogt_split = torch.load("data/embeddings/prepared_data_v7_saprot1.3b_seqonly_ogt_split.pt", map_location="cpu", weights_only=False)
    data['val_ogt'] = ogt_split['val_ogt']

    val_ogt_emb, val_ogt_seq, val_ogt_lbl, val_ogt_tmhmm = sanitize_data(data['val_ogt'], is_tm=False)
    val_ogt_e, val_ogt_aux = enrich_inputs(val_ogt_emb, val_ogt_seq, val_ogt_tmhmm)

    print(f"Total val_ogt sequences: {len(val_ogt_lbl)}")
    print(f"Target OGT distribution | Min: {val_ogt_lbl.min():.1f}°C, Max: {val_ogt_lbl.max():.1f}°C, Mean: {val_ogt_lbl.mean():.1f}°C")

    # Load 5-seed V9 ensemble models for OGT
    models = []
    for seed in range(1, 6):
        seed_dir = os.path.join(results_dir, f"seed{seed}")
        pt_ogt = os.path.join(seed_dir, "model_ogt.pt")
        m = MultiHeadSaProtV8().to(device)
        m.load_state_dict(torch.load(pt_ogt, map_location=device, weights_only=False))
        m.eval()
        models.append(m)

    val_ds = ProteinDataset(val_ogt_e, val_ogt_aux, val_ogt_lbl)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for emb, aux, y, _ in val_loader:
            emb, aux = emb.to(device), aux.to(device)
            seed_preds = []
            for m in models:
                pred_z = m(emb, aux, head='ogt')
                pred_raw = (pred_z * ogt_std) + ogt_mean
                seed_preds.append(pred_raw.cpu())
            ens_pred = torch.stack(seed_preds, dim=0).mean(dim=0)
            all_preds.append(ens_pred)
            all_targets.append(y)

    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    # 1. Raw MAE
    raw_mae = mean_absolute_error(targets, preds)

    # 2. De-quantized MAE (target jitter 0.5C on labels)
    np.random.seed(42)
    jittered_targets = targets + np.random.normal(0, 0.5, size=targets.shape)
    jitter_mae = mean_absolute_error(jittered_targets, preds)

    # 3. Bin-Balanced MAE (subsampling mesophiles 14% to balance temperature distribution)
    bins = list(range(0, 106, 5))
    weights = compute_bin_weights(torch.tensor(targets), bins, 0.3, 22.0, 0.75).numpy()
    weighted_mae = np.sum(weights * np.abs(targets - preds)) / np.sum(weights)

    # 4. Thermophilic Sub-group MAE (OGT > 50°C)
    thermo_mask = targets > 50.0
    thermo_mae = mean_absolute_error(targets[thermo_mask], preds[thermo_mask])

    print("\n=============================================")
    print("VAL OGT CLEANING & EVALUATION EXPERIMENT")
    print("=============================================")
    print(f"1. Raw Un-cleaned Val OGT MAE:            {raw_mae:.4f}°C")
    print(f"2. De-quantized (Jittered) Val OGT MAE:   {jitter_mae:.4f}°C")
    print(f"3. Bin-Balanced (Debiased) Val OGT MAE:   {weighted_mae:.4f}°C")
    print(f"4. Thermophilic Sub-group (>50°C) MAE:     {thermo_mae:.4f}°C")

if __name__ == "__main__":
    evaluate_val_ogt()
