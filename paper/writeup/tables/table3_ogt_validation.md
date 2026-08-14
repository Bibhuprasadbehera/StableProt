# Table 3: Optimal Growth Temperature (OGT) Environmental Regularizer Precision

| Model Architecture / Baseline | Representation Backbone | OGT MAE (°C) $\downarrow$ | OGT RMSE (°C) $\downarrow$ | Pearson ($r$) $\uparrow$ | Spearman ($\rho$) $\uparrow$ |
|:---|:---|:---:|:---:|:---:|:---:|
| **PRIME** | ESM-1b Regressor | 7.82 | 10.45 | 0.741 | 0.682 |
| **ThermoFormer** | Transformer Regressor | 7.64 | 10.18 | 0.755 | 0.698 |
| **StableProt OGT Head** | Disjoint SaProt 3Di MLP | **6.12** | **8.74** | **0.824** | **0.781** |
