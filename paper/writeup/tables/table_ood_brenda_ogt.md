# Out-Of-Distribution (OOD) BRENDA OGT Benchmark Comparative Results

Evaluation of StableProt V7 vs State-Of-The-Art baselines on strictly decontaminated out-of-distribution enzyme optimal growth temperatures from BRENDA (<40% sequence identity to training data, N=525).

| Model | MAE (°C) | RMSE (°C) | Pearson (r) | Spearman (ρ) |
| :--- | :---: | :---: | :---: | :---: |
| **StableProt V7 (Ours)** | 12.05 | 15.19 | 0.841 | 0.819 |
| **PRIME (AI4Protein/Prime_690M)** | 6.75 | 9.40 | 0.934 | 0.933 |
| **ThermoFormer (GinnM/ThermoFormer)** | 6.48 | 9.13 | 0.938 | 0.934 |

