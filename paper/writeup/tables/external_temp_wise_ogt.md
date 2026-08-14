# Temperature-Wise OGT Error and Uncertainty (External BRENDA OOD)

Bins left-closed [l, l+10). N = 525.

Calibration factor c = 1.18 (out-of-fold). Mean raw sigma = 11.42 C; mean calibrated sigma = 13.45 C (95% predictive interval approx +/- 26.4 C).

Point MAE and CRPS are comparable across models; CRPS of a deterministic forecast equals its MAE.
Int-MAE is reference-only and not comparable across models.

| Bin | Range | Count | StableProt V10 — point MAE (no uncertainty) | StableProt V10 — CRPS, calibrated (mean σ = 13.4 °C, c = 1.18) | StableProt V10 — Int-MAE at k=1 (mean σ = 11.4 °C, uncalibrated) | StableProt V10 — mean calibrated σ per bin (interval half-width) | StableProt V9 — point MAE | PRIME | ThermoFormer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0-10 | [0, 10) | 14 | 25.75 | 18.50 | 13.68 | 14.06 | 25.17 | 20.86 | 21.03 |
| 10-20 | [10, 20) | 31 | 19.04 | 12.49 | 6.14 | 15.94 | 17.89 | 8.80 | 8.47 |
| 20-30 | [20, 30) | 146 | 12.17 | 8.36 | 2.95 | 14.66 | 11.16 | 3.68 | 3.38 |
| 30-40 | [30, 40) | 35 | 10.64 | 7.45 | 1.98 | 15.32 | 9.50 | 2.26 | 2.35 |
| 40-50 | [40, 50) | 17 | 7.91 | 5.38 | 0.64 | 12.83 | 8.42 | 12.72 | 11.48 |
| 50-60 | [50, 60) | 59 | 9.41 | 6.54 | 1.90 | 13.11 | 10.70 | 12.51 | 12.19 |
| 60-70 | [60, 70) | 64 | 10.61 | 7.58 | 2.90 | 13.88 | 12.17 | 7.56 | 6.91 |
| 70-80 | [70, 80) | 73 | 8.16 | 5.89 | 1.57 | 12.62 | 9.54 | 5.30 | 5.27 |
| 80-90 | [80, 90) | 59 | 7.98 | 5.53 | 1.74 | 10.88 | 8.13 | 6.67 | 6.49 |
| 90-100 | [90, 100) | 27 | 6.53 | 4.57 | 1.24 | 9.25 | 5.15 | 5.25 | 5.29 |
