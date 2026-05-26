| Model Iteration | Architectural Sub-Type | MAE (°C) | PCC ($r$) | $R^2$ | MCC | F1 Score | ROC AUC | MAPE (%) | Top-10% Enrich Precision |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **TemStaPro (V0 Original)** | Pre-trained Binary Proxy Ensemble | 21.01 | 0.47 | -1.53 | 0.299 | 0.27 | 0.63 | 31.9% | 0.594 |
| **V1 Baseline** | Retrained Binary Proxy Ensemble | 28.14 | 0.48 | -3.10 | 0.249 | 0.20 | 0.60 | 44.0% | 0.594 |
| **V4 Improved Regr.** | Residual Continuous OGT Proxy | 26.32 | 0.46 | -2.65 | 0.241 | 0.19 | 0.59 | 41.0% | 0.594 |
| **V5 Multi-Head (ProtT5)** | Dedicated $T_m$ Head (ProtT5) | 12.62 | 0.50 | -0.18 | 0.303 | 0.30 | 0.62 | 18.9% | 0.594 |
| **V6 Multi-Head (ESM-2)** | Dedicated $T_m$ Head (ESM-2 3B) | 12.91 | 0.44 | -0.22 | 0.295 | 0.32 | 0.62 | 19.4% | 0.531 |
| **V6 Multi-Head (SaProt)** | **Dedicated $T_m$ Head (SaProt)** | **12.47** | **0.41** | **-0.17** | **0.253** | **0.36** | **0.62** | **19.3%** | **0.469** |
| **TemBERTure** | External Reference (Fine-Tuned) | 12.70 | 0.37 | -0.16 | 0.222 | 0.36 | 0.60 | 19.6% | 0.562 |
| **ESMStabP** | External Reference (Dedicated SOTA) | 14.85 | 0.33 | -0.51 | 0.150 | 0.33 | 0.58 | 22.4% | 0.375 |
| **DeepSTABp** | External Reference (Dedicated SOTA) | 13.51 | 0.44 | -0.30 | 0.288 | 0.30 | 0.60 | 20.5% | 0.531 |
| **ThermoFormer** | External Reference (Transformer SOTA) | 28.45 | 0.55 | -3.19 | 0.299 | 0.27 | 0.64 | 44.9% | 0.594 |
