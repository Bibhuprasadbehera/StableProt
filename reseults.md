BINNED MAE PERFORMANCE (Error in °C) — 210K OGT Test Set
---------------------------------------------------------------------------------------------------------
Model      | 0-10°C | 10-20°C | 20-30°C | 30-40°C | 40-50°C | 50-60°C | 60-70°C | 70-80°C | 80-90°C | 90-100°C | Overall
---------------------------------------------------------------------------------------------------------
V0 (Binary)| 33.1   | 23.6    | 11.5    | 7.5     | 6.6     | 6.2     | 4.7     | 9.1     | 19.2    | 30.0     | —
V1 (Binary)| 23.6   | 14.0    | 3.2     | 4.4     | 8.5     | 10.2    | 10.6    | 9.5     | 10.9    | 12.1     | —
V2 (Binary)| 9.1    | 10.6    | 9.9     | 12.2    | 13.5    | 12.6    | 10.6    | 9.1     | 7.9     | 6.4      | —
V3 (Regr.) | 24.9   | 15.5    | 4.0     | 4.0     | 7.8     | 9.7     | 10.3    | 9.5     | 11.0    | 12.9     | 5.664
V4 (Impr.) | 24.8   | 15.4    | 3.6     | 3.9     | 8.3     | 10.6    | 11.2    | 10.2    | 11.5    | 12.8     | 5.724
V5 (Multi) | 24.9   | 15.5    | 3.7     | 3.5     | 8.9     | 11.1    | 11.2    | 10.5    | 12.1    | 14.0     | 5.774
---------------------------------------------------------------------------------------------------------

NOTES:
- V0-V2: Binary classifiers, MAE computed via Expected Value conversion
- V0: Pre-trained models (thresholds 40-80°C only)
- V1: Retrained binary (thresholds 5-95°C)
- V2: 20 specialized binary models with balanced sampling
- V3: Single-head ProtT5 regression (MSE loss)
- V4: Improved regression (MSE loss, residual connection, cosine annealing scheduler, gradient clipping)
- V5: Multi-head ProtT5 model (Shared backbone, distinct OGT and Tm heads trained simultaneously)
- Continuous regression models (V3-V5) achieve strong mesophilic precision (~3.5-4.0°C MAE at 20-40°C) but struggle at extreme bounds compared to specialized binary ensemble V2.

ENSEMBLE MAE ON 210K OGT TEST (5-seed):
  V3: 5.664 °C
  V4: 5.724 °C
  V5 (OGT Head): 5.774 °C

V5 MULTI-HEAD TM TEST PERFORMANCE (ProThermDB test set):
  Ensemble MAE: 7.290 °C
  Per-seed MAE: 7.031, 7.365, 7.450, 7.473, 7.458 °C

PER-SEED MAE ON OGT:
  V3: 5.809, 5.723, 5.722, 5.707, 5.706 °C
  V4: 5.806, 5.760, 5.898, 5.698, 5.739 °C
