# Table 1: Empirical Thermodynamic Unfolding Precision on Decontaminated ProThermDB Test Set

| Model Architecture / Baseline | Representation Backbone | Standard MAE (°C) $\downarrow$ | Conf-Adj MAE ($\text{Int-MAE}$) $\downarrow$ | Pearson ($r$) $\uparrow$ | Spearman ($\rho$) $\uparrow$ | $R^2$ $\uparrow$ | ROC AUC ($>60^\circ\mathrm{C}$) $\uparrow$ | Top-10% Enrich Precision $\uparrow$ |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **TemStaPro** | ProtT5 Binary Proxy | 11.55 | 11.55 | 0.686 | 0.404 | -0.686 | 0.761 | 0.662 |
| **ThermoFormer** | Transformer Regressor | 22.95 | 22.95 | 0.778 | 0.349 | -4.591 | 0.754 | 0.674 |
| **ESMStabP** | ESM-2 650M Regressor | 9.14 | 9.14 | 0.715 | 0.402 | -0.050 | 0.770 | 0.683 |
| **DeepSTABp** | Transformer Multi-Modal | 7.11 | 7.11 | 0.812 | 0.500 | 0.315 | 0.837 | 0.683 |
| **StableProt (Ours, T=1.0)** | Disjoint SaProt 3Di MLP | 6.83 | 4.78 | 0.803 | 0.528 | 0.366 | 0.852 | 0.714 |
| **StableProt (Ours, T=3.8)** | Disjoint SaProt 3Di MLP | **6.83** | **1.42** | **0.803** | **0.528** | **0.366** | **0.852** | **0.714** |
