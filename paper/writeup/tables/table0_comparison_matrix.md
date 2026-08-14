# Table 0. Predictors evaluated in this study

| | TemStaPro | DeepSTABp | TemBERTure | ESMStabP | ThermoFormer | Pro-PRIME | **StableProt** (this work) |
|:--|:--|:--|:--|:--|:--|:--|:--|
| **Year** | 2024 | 2023 | 2024 | 2025 | 2024 | 2023 | — |
| **Backbone** | ProtT5-XL | ESM-1b | ProtBERT | ESM-2 650M | ThermoFormer encoder | ESM-1b | SaProt 650M |
| **Backbone parameters** | 3 B | 650 M | 420 M | 650 M | not reported | 690 M | 650 M |
| **Input** | Sequence | Sequence + experimental metadata | Sequence | Sequence | Sequence | Sequence + taxonomy | Sequence + predicted 3Di structure |
| **Structural information** | No | No | No | No | No | No | **Yes** |
| **Predicts** | $T_m$ threshold class | $T_m$ | $T_m$ | $T_m$ | OGT; $T_m$ from a separate checkpoint | OGT | **$T_m$ and OGT** |
| **Task coupling** | Single task | Shared parameters | Single task | Single task | Shared parameters | Single task | **Disjoint per-task heads** |
| **Output** | Six binary brackets | Point | Point | Point | Point | Point | **Mean and variance** |
| **Per-prediction uncertainty** | None | None | None | None | None | None | **Yes** |
| **ProThermDB MAE (°C)** | 11.55 ‡ | 7.11 | **5.76** | 9.14 | pending † | n/a § | 6.18 |
| **ProThermDB CRPS (°C)** | 11.55 ‡ | 7.11 | 5.76 | 9.14 | pending † | n/a § | **4.51** |
| **FireProtDB MAE (°C)** | 21.06 ‡ | 13.59 | 12.76 | 14.91 | pending † | n/a § | **11.92** |
| **FireProtDB CRPS (°C)** | 21.06 ‡ | 13.59 | 12.76 | 14.91 | pending † | n/a § | **8.79** |
| **Checkpoint evaluated** | *verify* | *verify* | *verify* | *verify* | `GinnM/ThermoFormer-TM` | `AI4Protein/Prime_690M` | `v9_disjoint`, five-seed ensemble |

Lower is better for both error columns; best value in each row is in bold. CRPS (continuous
ranked probability score) of a point forecast is exactly its mean absolute error, so the six
point predictors are scored on their own published output with nothing imputed; StableProt is
scored as a Gaussian predictive distribution with a single global variance scale fitted on
held-out data.

‡ TemStaPro returns threshold classes rather than a temperature; its error is a bracket-midpoint
proxy and is not directly comparable with the regression models.
† Pending re-evaluation with the dedicated $T_m$ checkpoint.
§ Pro-PRIME predicts organismal growth temperature only and is evaluated in the OGT tables.
