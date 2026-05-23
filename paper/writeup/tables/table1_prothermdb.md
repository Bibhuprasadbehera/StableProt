| Model Iteration | Architectural Sub-Type | MAE (°C) | PCC ($r$) | $R^2$ | MCC | F1 Score | ROC AUC | MAPE (%) | Top-10% Enrich Precision |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **TemStaPro (V0 Original)** | Pre-trained Binary Proxy Ensemble | 12.618 | 0.741 | -0.508 | 0.706 | 0.750 | 0.798 | 21.0% | 0.505 |
| **V2 Improved** | Specialized Binary Proxy Ensemble | 24.220 | 0.724 | -4.423 | 0.699 | 0.748 | 0.784 | 43.3% | 0.524 |
| **V3 Regression** | Continuous Single-Head OGT Proxy | 16.715 | 0.745 | -1.457 | 0.669 | 0.707 | 0.784 | 28.5% | 0.513 |
| **V4 Improved** | Residual Continuous OGT Proxy | 17.261 | 0.778 | -1.550 | 0.679 | 0.717 | 0.787 | 29.7% | 0.511 |
| **TemBERTure** | External Reference (Fine-Tuned) | 8.350 | 0.743 | 0.228 | 0.506 | 0.681 | 0.860 | 14.8% | 0.681 |
| **ESMStabP** | External Reference (Dedicated SOTA) | 6.420 | 0.830 | 0.546 | 0.596 | 0.735 | 0.900 | 11.3% | 0.760 |
| **V5 Multi-Head** | **Dedicated $T_m$ Head (ProtT5)** | **7.290** | **0.836** | **0.444** | **0.711** | **0.753** | **0.875** | **12.5%** | **0.557** |
| **V6 Multi-Head** | **Dedicated $T_m$ Head (ESM-2 3B)** | **5.748** | **0.872** | **0.644** | **0.736** | **0.785** | **0.887** | **9.6%** | **0.681** |
