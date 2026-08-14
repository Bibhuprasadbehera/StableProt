# Out-Of-Distribution (OOD) BRENDA OGT Benchmark Comparative Results

Evaluation of StableProt V9 vs State-Of-The-Art baselines on strictly decontaminated out-of-distribution enzyme optimal growth temperatures from BRENDA (<40% sequence identity to training data, N=525).

| Model | MAE (°C) | RMSE (°C) | Pearson (r) | Spearman (ρ) |
| :--- | :---: | :---: | :---: | :---: |
| **StableProt (v9_disjoint) (Ours)** | 10.93 | 13.97 | 0.854 | 0.838 |
| **StableProt (v9_disjoint) (Int-MAE, k=1)** | 8.50 | 11.96 | 0.854 | 0.838 |
| **StableProt (v9_disjoint) (Int-MAE, calibrated c=5.64)** | 2.59 | 6.33 | 0.854 | 0.838 |
| **PRIME (AI4Protein/Prime_690M)** | 6.75 | 9.40 | 0.934 | 0.933 |
| **ThermoFormer (GinnM/ThermoFormer)** | 6.48 | 9.13 | 0.938 | 0.934 |

