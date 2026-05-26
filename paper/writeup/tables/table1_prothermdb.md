| Model Iteration | Architectural Sub-Type | MAE (°C) | PCC ($r$) | $R^2$ | MCC | F1 Score | ROC AUC | MAPE (%) | Top-10% Enrich Precision |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **TemStaPro (V0 Original)** | Pre-trained Binary Proxy Ensemble | 12.62 | 0.74 | -0.51 | 0.706 | 0.75 | 0.80 | 21.0% | 0.505 |
| **V1 Baseline** | Retrained Binary Proxy Ensemble | 18.77 | 0.78 | -1.96 | 0.666 | 0.70 | 0.79 | 32.2% | 0.493 |
| **V2 Improved** | Specialized Binary Proxy Ensemble | 24.22 | 0.72 | -4.42 | 0.699 | 0.75 | 0.78 | 43.3% | 0.524 |
| **V3 Regression** | Continuous Single-Head OGT Proxy | 16.72 | 0.74 | -1.46 | 0.669 | 0.71 | 0.78 | 28.5% | 0.513 |
| **V4 Improved Regr.** | Residual Continuous OGT Proxy | 17.26 | 0.78 | -1.55 | 0.679 | 0.72 | 0.79 | 29.7% | 0.511 |
| **V5 Multi-Head** | Dedicated $T_m$ Head (ProtT5) | 7.29 | 0.84 | 0.44 | 0.711 | 0.75 | 0.88 | 12.5% | 0.557 |
| **V6 Multi-Head** | Dedicated $T_m$ Head (ESM-2 3B) | **5.53** | **0.85** | **0.63** | **0.710** | **0.76** | **0.87** | **9.4%** | **0.668** |
| **V7 ESM-2 (Mode D2)** | Transfer Learning Clean (ESM-2) | 7.21 | 0.77 | 0.43 | 0.608 | 0.72 | 0.85 | 12.7% | 0.505 |
| **V7 ESM-2 Joint** | **Weighted Multi-Task ESM-2** | **7.21** | **0.82** | **0.41** | **0.655** | **0.74** | **0.86** | **12.6%** | **0.637** |
| **TemBERTure** | External Reference (Fine-Tuned) | 5.49 | 0.86 | 0.66 | 0.715 | 0.77 | 0.87 | 9.4% | 0.716 |
| **ESMStabP** | External Reference (Dedicated SOTA) | 8.84 | 0.76 | 0.21 | 0.540 | 0.67 | 0.81 | 15.3% | 0.559 |
| **DeepSTABp** | External Reference (Dedicated SOTA) | 6.80 | 0.83 | 0.50 | 0.708 | 0.76 | 0.85 | 11.6% | 0.562 |
| **ThermoFormer** | External Reference (Transformer SOTA) | 22.16 | 0.80 | -3.07 | 0.715 | 0.76 | 0.80 | 39.0% | 0.524 |
