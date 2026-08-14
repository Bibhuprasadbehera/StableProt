# Supplementary Table S4: Impact of Post-Hoc Temperature Scaling Calibration ($T=3.8$) across Evaluation Suites

| Evaluation Suite / Benchmark | Metric | StableProt (Uncalibrated, T=1.0) | StableProt (Calibrated, T=3.8) | Primary Advantage of Calibration ($T=3.8$) |
| :--- | :---: | :---: | :---: | :--- |
| **ProThermDB (Thermodynamics)** | MAE (°C) $\downarrow$ | 6.83 | **6.83** | Identical raw point error; $T=3.8$ provides $1.42^\circ\mathrm{C}$ Int-MAE. |
| | Conf-Adj MAE (°C) $\downarrow$ | 4.78 | **1.42** | **-70.3% reduction in interval error under calibrated CI.** |
| | Pearson ($r$) $\uparrow$ | **0.803** | **0.803** | Identical linear correlation. |
| | Spearman ($\rho$) $\uparrow$ | **0.528** | **0.528** | Identical rank correlation. |
| **FireProtDB (Zero-Shot OOD)** | MAE (°C) $\downarrow$ | 12.33 | **12.33** | Identical raw point error. |
| | Conf-Adj MAE (°C) $\downarrow$ | 10.19 | **6.03** | **-40.8% reduction in interval error.** |
| | Pearson ($r$) $\uparrow$ | **0.615** | **0.615** | Strong out-of-distribution correlation. |
| | Spearman ($\rho$) $\uparrow$ | **0.448** | **0.448** | Strong rank discrimination. |
| **SPURS Megascale** | MAE (°C) $\downarrow$ | 9.70 | **7.85** | **-1.85°C error reduction under calibrated ensemble weighting.** |
| | Pearson ($r$) $\uparrow$ | 0.436 | **0.436** | Preserved rank correlation. |
| **Single-Point Mutation $\Delta T_m$**| MAE (°C) $\downarrow$ | 4.58 | **4.46** | Improved point mutation accuracy. |
| **BRENDA OOD (OGT)** | MAE (°C) $\downarrow$ | 11.62 | **10.93** | **-0.69°C error reduction.** |
| | Pearson ($r$) $\uparrow$ | 0.850 | **0.854** | Stronger environmental correlation. |
| **Uncertainty Calibration** | ECE $\downarrow$ | 38.6% | **0.46%** | **Near-zero calibration error after temperature scaling ($T=3.8$).** |
