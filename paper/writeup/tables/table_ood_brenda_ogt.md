# Out-Of-Distribution (OOD) BRENDA OGT Benchmark Comparative Results

Evaluation of StableProt V9 vs State-Of-The-Art baselines on strictly decontaminated out-of-distribution enzyme optimal growth temperatures from BRENDA (<40% sequence identity to training data, N=525).

| Model | MAE (°C) | RMSE (°C) | Pearson (r) | Spearman (ρ) |
| :--- | :---: | :---: | :---: | :---: |
| **StableProt V9 (Ours)** | 11.62 | 14.75 | 0.850 | 0.833 |
| **StableProt V9 (Conf-Adj, T=1.0)** | 8.33 | 11.94 | 0.850 | 0.833 |
| **StableProt V9 (Calibrated Conf-Adj, T=3.8)** | 2.97 | 6.79 | 0.850 | 0.833 |
| **PRIME (AI4Protein/Prime_690M)** | 6.75 | 9.40 | 0.934 | 0.933 |
| **ThermoFormer (GinnM/ThermoFormer)** | 6.48 | 9.13 | 0.938 | 0.934 |

