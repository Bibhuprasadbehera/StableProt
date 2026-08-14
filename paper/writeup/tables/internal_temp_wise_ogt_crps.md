# Probabilistic Accuracy (CRPS) by Temperature Bin (Internal BacDive Test)

Bins left-closed [l, l+10). N = 4854. StableProt uses its predictive distribution (sigma scale c fitted out-of-fold, c=0.87 for V10). PRIME and ThermoFormer emit no interval, so their CRPS equals their MAE.

| Bin | n | StableProt V10 (Ours) | StableProt V9 | PRIME | ThermoFormer |
| --- | --- | --- | --- | --- | --- |
| 0-10 | 1 | 11.72 | 12.08 | 22.93 | 22.57 |
| 10-20 | 32 | 8.16 | 8.52 | 11.03 | 10.53 |
| 20-30 | 1736 | 5.46 | 5.42 | 2.09 | 2.05 |
| 30-40 | 1675 | 5.88 | 5.72 | 2.35 | 2.28 |
| 40-50 | 392 | 5.73 | 6.33 | 10.99 | 10.62 |
| 50-60 | 532 | 4.96 | 5.87 | 12.19 | 11.51 |
| 60-70 | 250 | 6.24 | 7.32 | 12.43 | 11.29 |
| 70-80 | 133 | 5.57 | 6.23 | 9.84 | 8.66 |
| 80-90 | 73 | 4.52 | 4.87 | 7.28 | 6.60 |
| 90-100 | 30 | 7.19 | 7.61 | 12.62 | 12.00 |
