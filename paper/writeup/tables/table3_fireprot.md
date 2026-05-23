| Model Iteration | Architectural Sub-Type | MAE (°C) | PCC ($r$) | $R^2$ | MCC | F1 Score | ROC AUC | MAPE (%) | Top-10% Enrich Precision |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **TemStaPro (V0 Original)** | Pre-trained Binary Proxy Ensemble | 20.86 | 0.43 | -1.74 | 0.286 | 0.26 | 0.63 | 31.6% | 0.500 |
| **V2 Improved** | Specialized Binary Proxy Ensemble | 32.67 | 0.45 | -5.18 | 0.288 | 0.28 | 0.62 | 52.4% | 0.500 |
| **V3 Regression** | Continuous Single-Head OGT Proxy | 26.01 | 0.41 | -2.93 | 0.233 | 0.20 | 0.61 | 40.4% | 0.500 |
| **V4 Improved** | Residual Continuous OGT Proxy | 26.38 | 0.44 | -3.01 | 0.218 | 0.16 | 0.60 | 41.0% | 0.500 |
| **TemBERTure** | External Reference (Fine-Tuned) | 8.85 | 0.80 | 0.41 | 0.616 | 0.81 | 0.88 | 14.9% | 0.667 |
| **ESMStabP** | External Reference (Dedicated SOTA) | 7.92 | 0.85 | 0.56 | 0.573 | 0.78 | 0.88 | 13.5% | 0.643 |
| **V5 Multi-Head** | **Dedicated $T_m$ Head (ProtT5)** | **12.59** | **0.45** | **-0.27** | **0.288** | **0.28** | **0.62** | **18.8%** | **0.500** |
| **V6 Multi-Head** | **Dedicated $T_m$ Head (ESM-2 3B)** | **5.82** | **0.89** | **0.75** | **0.672** | **0.83** | **0.93** | **9.8%** | **0.786** |
