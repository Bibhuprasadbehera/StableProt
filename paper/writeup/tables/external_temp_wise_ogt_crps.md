# Probabilistic Accuracy (CRPS) by Temperature Bin (External BRENDA OOD)

Bins left-closed [l, l+10). N = 525. StableProt uses its predictive distribution (sigma scale c fitted out-of-fold, c=1.18 for V10). PRIME and ThermoFormer emit no interval, so their CRPS equals their MAE.

| Bin | n | StableProt V10 (Ours) | StableProt V9 | PRIME | ThermoFormer |
| --- | --- | --- | --- | --- | --- |
| 0-10 | 14 | 18.50 | 17.49 | 20.86 | 21.03 |
| 10-20 | 31 | 12.49 | 13.28 | 8.80 | 8.47 |
| 20-30 | 146 | 8.36 | 8.72 | 3.68 | 3.38 |
| 30-40 | 35 | 7.45 | 7.64 | 2.26 | 2.35 |
| 40-50 | 17 | 5.38 | 5.83 | 12.72 | 11.48 |
| 50-60 | 59 | 6.54 | 7.75 | 12.51 | 12.19 |
| 60-70 | 64 | 7.58 | 8.74 | 7.56 | 6.91 |
| 70-80 | 73 | 5.89 | 7.12 | 5.30 | 5.27 |
| 80-90 | 59 | 5.53 | 5.73 | 6.67 | 6.49 |
| 90-100 | 27 | 4.57 | 3.66 | 5.25 | 5.29 |
