# Temperature-Wise OGT Error and Uncertainty (Internal BacDive Test)

Bins left-closed [l, l+10). N = 4854.

Calibration factor c = 0.87 (out-of-fold). Mean raw sigma = 11.24 C; mean calibrated sigma = 9.82 C (95% predictive interval approx +/- 19.2 C).

Point MAE and CRPS are comparable across models; CRPS of a deterministic forecast equals its MAE.
Int-MAE is reference-only and not comparable across models.

| Bin | Range | Count | StableProt V10 — point MAE (no uncertainty) | StableProt V10 — CRPS, calibrated (mean σ = 9.8 °C, c = 0.87) | StableProt V10 — Int-MAE at k=1 (mean σ = 11.2 °C, uncalibrated) | StableProt V10 — mean calibrated σ per bin (interval half-width) | StableProt V9 — point MAE | PRIME | ThermoFormer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0-10 | [0, 10) | 1 | 15.15 | 11.72 | 8.11 | 6.13 | 15.78 | 22.93 | 22.57 |
| 10-20 | [10, 20) | 32 | 12.21 | 8.16 | 2.44 | 10.03 | 12.12 | 11.03 | 10.53 |
| 20-30 | [20, 30) | 1736 | 7.79 | 5.46 | 1.20 | 9.56 | 7.24 | 2.09 | 2.05 |
| 30-40 | [30, 40) | 1675 | 8.16 | 5.88 | 1.46 | 10.35 | 7.68 | 2.35 | 2.28 |
| 40-50 | [40, 50) | 392 | 8.16 | 5.73 | 1.39 | 9.68 | 8.47 | 10.99 | 10.62 |
| 50-60 | [50, 60) | 532 | 6.98 | 4.96 | 0.83 | 9.86 | 7.94 | 12.19 | 11.51 |
| 60-70 | [60, 70) | 250 | 8.90 | 6.24 | 1.60 | 9.75 | 9.92 | 12.43 | 11.29 |
| 70-80 | [70, 80) | 133 | 7.80 | 5.57 | 1.74 | 7.93 | 8.58 | 9.84 | 8.66 |
| 80-90 | [80, 90) | 73 | 6.37 | 4.52 | 1.00 | 8.06 | 6.55 | 7.28 | 6.60 |
| 90-100 | [90, 100) | 30 | 10.80 | 7.19 | 2.03 | 9.57 | 10.87 | 12.62 | 12.00 |
