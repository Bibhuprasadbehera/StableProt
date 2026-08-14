# Supplementary Table S2: Comprehensive Architectural Progression and Diagnostic Ablation Study

| Model Configuration & Ablation State | ProThermDB MAE (°C) | ProThermDB Int-MAE (°C) | FireProtDB MAE (°C) | FireProtDB Int-MAE (°C) | Architectural & Biophysical Impact |
|:---|:---:|:---:|:---:|:---:|:---|
| **Baseline MLP (Raw 1D Seq)** | 12.28 | 12.28 | 28.21 | 28.21 | Simple multi-layer perceptron on primary sequence proxy tokens without structural context or uncertainty bounding. |
| **Regularized MLP (Dropout 0.3/0.2)** | 10.60 | 10.60 | 32.47 | 32.47 | Added batch normalization and residual dropout (`0.3`/`0.2`), improving empirical interpolation but overfitting on zero-shot holdouts. |
| **Continuous Regressor (L1/L2)** | 9.38 | 9.38 | 25.96 | 25.96 | Shifted from binary classification proxy targets to direct continuous regression (`L1/L2` loss) across unfolding temperatures. |
| **Residual Regressor (Skip Connections)** | 8.16 | 8.16 | 26.39 | 26.39 | Introduced skip-connection residual projections (`residual_proj`), reducing ProThermDB MAE by $>1.2^\circ\mathrm{C}$. |
| **Auxiliary Bottleneck Projection (64-dim)** | 6.84 | 6.84 | 12.69 | 12.69 | Added dedicated 64-dim projection bottlenecks (`Linear(9,64)`) separating scalar features from high-dimensional token representations. |
| **Structure-Aware SaProt 3Di Tokens** | 6.11 | 6.11 | 10.84 | 10.84 | Integrated Foldseek 3Di conformational structural tokens (`1280-dim`), providing dramatic $>1.8^\circ\mathrm{C}$ zero-shot extremophile improvement. |
| **Shared Multi-Task Control (Gradient Conflict)** | 7.61 | 7.61 | 11.45 | 11.45 | **Negative Control**: Shared hidden layers for simultaneous $T_m$ and OGT optimization caused gradient interference ($\cos \theta = -0.077$, +1.50°C error). |
| **StableProt (Uncalibrated Pipeline, T=1.0)** | 6.83 | 4.78 | 12.33 | 10.19 | Disjoint alternating optimization + NLL confidence intervals + mesophilic subsampling (`14%`). |
| **StableProt (Calibrated Production, T=3.8)** | **6.83** | **1.42** | **12.33** | **6.03** | **Final production architecture with post-hoc calibration scaling ($T=3.8$) reducing ECE to 0.46%.** |
