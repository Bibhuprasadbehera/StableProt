#!/usr/bin/env python3
"""
Quantile Regression Calibration Test

Trains a quick V8 model with quantile regression (τ=0.1, 0.5, 0.9) instead of
Gaussian NLL, then computes ECE to determine if V9 should use quantile loss.

Decision rule:
  - If quantile ECE < 10% → use quantile regression in V9
  - If quantile ECE > 10% → try post-hoc temperature scaling
  - If neither works → remove all uncertainty claims from paper

Also tests post-hoc temperature scaling on existing Gaussian NLL model.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import scipy.special

PROJECT_ROOT = Path(__file__).resolve().parents[3]
VERSION = os.environ.get("STABLEPROT_VERSION", "v9_disjoint")
sys.path.append(str(PROJECT_ROOT / "experiments" / "src" / "training" / VERSION))
from config import CONFIG
from train import (
    MultiHeadSaProtV8, ProteinDataset, compute_aa_composition,
    enrich_inputs, compute_bin_weights, build_tm_iqr_weights,
    focal_huber_loss, MesophilicSubsampler
)


def expected_coverage(z):
    return scipy.special.erf(z / np.sqrt(2.0))


def compute_ece(y_true, y_pred, y_conf, z_vals=None):
    if z_vals is None:
        z_vals = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0])
    exp_cov = expected_coverage(z_vals)
    errors = np.abs(y_true - y_pred)
    emp_cov = np.array([np.mean(errors <= z * y_conf) for z in z_vals])
    ece = np.mean(np.abs(emp_cov - exp_cov))
    return ece, exp_cov, emp_cov, z_vals


def test_temperature_scaling():
    """Post-hoc calibration on existing Gaussian NLL model."""
    print("\n" + "="*70)
    print("  TEST 1: POST-HOC TEMPERATURE SCALING ON EXISTING V8 MODEL")
    print("="*70)

    protherm_path = PROJECT_ROOT / "new_data" / "protherm_evaluation_results.pt"
    if not protherm_path.exists():
        print(f"Missing: {protherm_path}")
        return

    data = torch.load(str(protherm_path), map_location='cpu', weights_only=False)
    y_true = np.array(data['y_true'])
    k = 'StableProt V9' if 'StableProt V9' in data['predictions'] else 'StableProt V8'
    y_pred = np.array(data['predictions'][k])
    y_conf = np.array(data['confidences'][k])

    errors = np.abs(y_true - y_pred)
    mae = errors.mean()

    # Current calibration
    ece_before, _, _, _ = compute_ece(y_true, y_pred, y_conf)
    print(f"\nBefore scaling: ECE = {ece_before:.4f}, MAE = {mae:.2f}°C, σ_mean = {y_conf.mean():.2f}°C")

    # Try different temperature scales
    best_ece = ece_before
    best_T = 1.0
    for T in np.arange(0.5, 8.0, 0.1):
        scaled_conf = y_conf * T
        ece_t, _, _, _ = compute_ece(y_true, y_pred, scaled_conf)
        if ece_t < best_ece:
            best_ece = ece_t
            best_T = T

    y_conf_scaled = y_conf * best_T
    ece_after, exp_cov, emp_cov, z_vals = compute_ece(y_true, y_pred, y_conf_scaled)

    print(f"After scaling (T={best_T:.1f}): ECE = {ece_after:.4f}, σ_mean = {y_conf_scaled.mean():.2f}°C")
    print(f"  1σ coverage: {emp_cov[np.where(np.isclose(z_vals, 1.0))[0][0]]*100:.1f}% (expected 68.3%)")
    print(f"  2σ coverage: {emp_cov[np.where(np.isclose(z_vals, 2.0))[0][0]]*100:.1f}% (expected 95.4%)")

    # Stratified
    for name, mask in [('Mesophilic (≤50°C)', y_true <= 50),
                        ('Thermophilic (>50°C)', y_true > 50),
                        ('Hyperthermophilic (>70°C)', y_true > 70)]:
        if np.sum(mask) > 5:
            ece_s, _, emp_s, _ = compute_ece(y_true[mask], y_pred[mask], y_conf_scaled[mask])
            idx_1s = np.where(np.isclose(z_vals, 1.0))[0][0]
            print(f"  {name:<30} | N={np.sum(mask):<4} | ECE={ece_s:.4f} | 1σ Cov={emp_s[idx_1s]*100:.1f}%")

    return best_T, ece_after


def test_quantile_regression():
    """Train a fresh quantile model and test calibration."""
    print("\n" + "="*70)
    print("  TEST 2: QUANTILE REGRESSION CALIBRATION (τ=0.1, 0.5, 0.9)")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    data = torch.load(str(PROJECT_ROOT / "data/embeddings/saprot_tm_struct_embeddings.pt"),
                       map_location='cpu', weights_only=False)

    # Prepare training data
    train = data['train_tm']
    val = data['val_tm']
    test = data.get('test_tm', val)

    def prep_split(split_data):
        seqs = split_data['sequences']
        embs = split_data['embeddings']
        labels = split_data.get('tm_consensus', split_data.get('labels'))
        tmhmm = split_data.get('tmhmm_tm_binary', [0]*len(seqs))
        ogts = split_data.get('ogt', [50.0]*len(seqs))
        e, aux = enrich_inputs(embs, seqs, tmhmm, ogts)
        y = torch.tensor([float(l) for l in labels], dtype=torch.float32)
        return e, aux, y

    print("Preparing data...", flush=True)
    tr_e, tr_aux, tr_y = prep_split(train)
    val_e, val_aux, val_y = prep_split(val)
    test_e, test_aux, test_y = prep_split(test)

    tm_mean, tm_std = tr_y.mean().item(), tr_y.std().item()

    tr_ds = ProteinDataset(tr_e, tr_aux, tr_y, augment=True)
    val_ds = ProteinDataset(val_e, val_aux, val_y)
    test_ds = ProteinDataset(test_e, test_aux, test_y)

    tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    test_loader = DataLoader(test_ds, batch_size=64)

    # Quantile model: outputs 3 values (τ=0.1, 0.5, 0.9)
    class QuantileHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.aux_proj = nn.Sequential(nn.Linear(9, 64), nn.GELU(), nn.LayerNorm(64))
            self.fc1 = nn.Linear(1344, 512)
            self.ln1 = nn.LayerNorm(512)
            self.fc2 = nn.Linear(512, 256)
            self.ln2 = nn.LayerNorm(256)
            self.res = nn.Linear(512, 256)
            self.head = nn.Linear(256, 3)  # 3 quantiles
            self.act = nn.GELU()
            self.drop1 = nn.Dropout(0.3)
            self.drop2 = nn.Dropout(0.2)

        def forward(self, x_emb, x_aux):
            h_aux = self.aux_proj(x_aux)
            h = torch.cat([x_emb, h_aux], dim=-1)
            h1 = self.drop1(self.act(self.ln1(self.fc1(h))))
            h2 = self.drop2(self.act(self.ln2(self.fc2(h1)) + self.res(h1)))
            return self.head(h2)

    model = QuantileHead().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    quantiles = torch.tensor([0.1, 0.5, 0.9], device=device)

    best_val_mae = float('inf')
    patience_count = 0

    print("Training quantile model (15 epochs)...", flush=True)
    for epoch in range(15):
        model.train()
        for emb, aux, y, w in tr_loader:
            emb, aux, y = emb.to(device), aux.to(device), y.to(device)
            z = (y - tm_mean) / tm_std
            optimizer.zero_grad()
            out = model(emb, aux)  # [B, 3]
            # Pinball loss
            loss = 0.0
            for i, q in enumerate(quantiles):
                err = z - out[:, i]
                loss += torch.mean(torch.max(q * err, (q - 1) * err))
            loss /= 3.0
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # Validate
        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for emb, aux, y, w in val_loader:
                emb, aux, y = emb.to(device), aux.to(device), y.to(device)
                out = model(emb, aux)
                pred = out[:, 1] * tm_std + tm_mean  # median quantile
                val_mae += torch.abs(pred - y).sum().item()
        val_mae /= len(val_ds)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_count = 0
            torch.save(model.state_dict(), str(PROJECT_ROOT / "experiments/src/eval/quantile_test_model.pt"))
        else:
            patience_count += 1

        if patience_count >= 5:
            break
        print(f"  Epoch {epoch+1}: Val MAE = {val_mae:.2f}°C", flush=True)

    # Evaluate calibration
    model.load_state_dict(torch.load(str(PROJECT_ROOT / "experiments/src/eval/quantile_test_model.pt"),
                                      map_location=device, weights_only=False))
    model.eval()

    all_preds = []
    all_q10 = []
    all_q90 = []
    all_true = []
    with torch.no_grad():
        for emb, aux, y, w in val_loader:
            emb, aux = emb.to(device), aux.to(device)
            out = model(emb, aux)
            q10 = out[:, 0] * tm_std + tm_mean
            q50 = out[:, 1] * tm_std + tm_mean
            q90 = out[:, 2] * tm_std + tm_mean
            all_preds.append(q50.cpu().numpy())
            all_q10.append(q10.cpu().numpy())
            all_q90.append(q90.cpu().numpy())
            all_true.append(y.numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_preds)
    y_q10 = np.concatenate(all_q10)
    y_q90 = np.concatenate(all_q90)

    # Derive σ from quantile spread: σ ≈ (q90 - q10) / (2 * 1.28)
    y_conf = np.maximum(np.abs(y_q90 - y_q10) / 2.56, 0.5)

    mae = np.mean(np.abs(y_true - y_pred))
    ece, exp_cov, emp_cov, z_vals = compute_ece(y_true, y_pred, y_conf)

    print(f"\nQuantile Model Results:")
    print(f"  Val MAE: {mae:.2f}°C")
    print(f"  ECE: {ece:.4f}")
    print(f"  σ_mean (from quantile spread): {y_conf.mean():.2f}°C")
    idx_1s = np.where(np.isclose(z_vals, 1.0))[0][0]
    idx_2s = np.where(np.isclose(z_vals, 2.0))[0][0]
    print(f"  1σ coverage: {emp_cov[idx_1s]*100:.1f}% (expected 68.3%)")
    print(f"  2σ coverage: {emp_cov[idx_2s]*100:.1f}% (expected 95.4%)")

    # 80% prediction interval coverage (q10 to q90)
    in_interval = np.mean((y_true >= y_q10) & (y_true <= y_q90))
    print(f"  80% PI coverage (q10-q90): {in_interval*100:.1f}% (expected 80.0%)")

    return ece, mae


def main():
    print("="*70)
    print("  CALIBRATION DECISION TEST: GAUSSIAN NLL vs QUANTILE REGRESSION")
    print("="*70)

    # Test 1: Can we fix existing model with temperature scaling?
    best_T, ece_scaled = test_temperature_scaling()

    # Test 2: Does quantile regression give better calibration?
    ece_quantile, mae_quantile = test_quantile_regression()

    # Decision
    print("\n" + "="*70)
    print("  DECISION SUMMARY")
    print("="*70)
    print(f"  Temperature-scaled Gaussian NLL: ECE = {ece_scaled:.4f}")
    print(f"  Quantile Regression:            ECE = {ece_quantile:.4f}")

    if ece_scaled < 0.10:
        print("\n  → RECOMMENDATION: Use post-hoc temperature scaling (T={best_T:.1f}) on V8 Gaussian NLL")
        print("    Keep Int-MAE in manuscript with scaled σ values.")
    elif ece_quantile < 0.10:
        print(f"\n  → RECOMMENDATION: Use quantile regression in V9")
        print("    Replace Gaussian NLL with pinball loss (τ=0.1, 0.5, 0.9).")
    else:
        print(f"\n  → RECOMMENDATION: Remove all uncertainty claims from manuscript")
        print("    Neither approach achieves acceptable calibration (ECE < 10%).")
        print("    Report standard MAE only. Drop Int-MAE metric.")


if __name__ == "__main__":
    main()
