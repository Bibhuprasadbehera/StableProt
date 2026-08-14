# Table 2: Zero-Shot Out-of-Distribution Generalization on FireProtDB Holdout Set

| Model Architecture / Baseline | Representation Backbone | Standard MAE (°C) $\downarrow$ | Conf-Adj MAE ($\text{Int-MAE}$) $\downarrow$ | Pearson ($r$) $\uparrow$ | Spearman ($\rho$) $\uparrow$ | $R^2$ $\uparrow$ | ROC AUC ($>60^\circ\mathrm{C}$) $\uparrow$ |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **TemStaPro** | ProtT5 Binary Proxy | 16.42 | 16.42 | 0.412 | 0.288 | -1.105 | 0.612 |
| **ThermoFormer** | Transformer Regressor | 31.84 | 31.84 | 0.540 | 0.312 | -6.892 | 0.605 |
| **ESMStabP** | ESM-2 650M Regressor | 14.12 | 14.12 | 0.485 | 0.320 | -0.551 | 0.640 |
| **DeepSTABp** | Transformer Multi-Modal | 13.05 | 13.05 | 0.591 | 0.410 | -0.328 | 0.715 |
| **StableProt (Ours, T=1.0)** | Disjoint SaProt 3Di MLP | 12.33 | 10.19 | 0.615 | 0.448 | -0.186 | 0.738 |
| **StableProt (Ours, T=3.8)** | Disjoint SaProt 3Di MLP | **12.33** | **6.03** | **0.615** | **0.448** | **-0.186** | **0.738** |
