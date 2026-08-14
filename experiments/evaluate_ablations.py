import os
import sys
import glob
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

sys.path.append(os.path.abspath("experiments/src/model_calibration"))
from sweep_runner import CalibrationSaProtV8
sys.path.append(os.path.abspath("experiments/src/training/v9_disjoint"))
from train import enrich_inputs

def evaluate_checkpoint(ckpt_dir, val_tm_e, val_tm_aux, val_tm_lbl, emb_dim=1280):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    norm_path = os.path.join(ckpt_dir, "normalization_stats.pt")
    if os.path.exists(norm_path):
        norms = torch.load(norm_path, map_location='cpu', weights_only=False)
        tm_mean, tm_std = norms['tm_mean'], norms['tm_std']
    else:
        raise FileNotFoundError(f"normalization_stats.pt not found at {norm_path}")

    pt_tm = os.path.join(ckpt_dir, "model_tm.pt")
    if not os.path.exists(pt_tm):
        return None

    cfg = {
        'mlp_layers': 2,
        'hidden1': 512,
        'hidden2': 256,
        'norm_type': 'layernorm',
        'use_residuals': True,
        'dropout1': 0.3,
        'dropout2': 0.2,
        'aux_proj_dim': 64,
        'backbone_adapt_dim': 0,
        'pool_layer_type': 'none',
        'loss_tm_type': 'nll_softplus',
        'nll_softplus_eps': 1e-4
    }

    m_tm = CalibrationSaProtV8(emb_dim=emb_dim, config_dict=cfg).to(device)
    m_tm.load_state_dict(torch.load(pt_tm, map_location=device, weights_only=False))
    m_tm.eval()

    with torch.no_grad():
        e_tm, a_tm = val_tm_e.to(device), val_tm_aux.to(device)
        tm_pred_z, _ = m_tm(e_tm, a_tm, head='tm')
        tm_preds = (tm_pred_z.cpu().numpy() * tm_std) + tm_mean

    tm_true = val_tm_lbl.numpy()
    tm_mae = mean_absolute_error(tm_true, tm_preds)
    
    return {
        "tm_mae": round(tm_mae, 3)
    }

def main():
    print("Loading data/embeddings/prepared_data_v3.pt validation tensors...")
    data = torch.load("data/embeddings/prepared_data_v3.pt", map_location="cpu", weights_only=False)
    
    val_tm_dict = data['val_tm']
    val_tm_emb = val_tm_dict['embeddings'][:, :1280]  # First 1280 SaProt dims
    val_tm_seq = val_tm_dict['sequences']
    val_tm_lbl = val_tm_dict['labels']
    val_tm_tmhmm = val_tm_dict.get('tmhmm', None)
    val_tm_ogt = torch.full((len(val_tm_seq),), 37.51, dtype=torch.float32)

    val_tm_e, val_tm_aux = enrich_inputs(val_tm_emb, val_tm_seq, val_tm_tmhmm, val_tm_ogt)

    print("\n--- TARGET JITTER ABLATION RESULTS ---")
    jitter_dirs = sorted(glob.glob("experiments/src/model_calibration/checkpoints/data/target_jitter_std*"))
    for d in jitter_dirs:
        res = evaluate_checkpoint(d, val_tm_e, val_tm_aux, val_tm_lbl, emb_dim=1280)
        if res:
            name = os.path.basename(d)
            print(f"{name:<30} | Tm MAE: {res['tm_mae']}°C")

    print("\n--- IQR FILTER ABLATION RESULTS ---")
    iqr_dirs = sorted(glob.glob("experiments/src/model_calibration/checkpoints/data/iqr_filter_max*"))
    for d in iqr_dirs:
        res = evaluate_checkpoint(d, val_tm_e, val_tm_aux, val_tm_lbl, emb_dim=1280)
        if res:
            name = os.path.basename(d)
            print(f"{name:<30} | Tm MAE: {res['tm_mae']}°C")

if __name__ == "__main__":
    main()
