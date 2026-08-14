# Table 5: Prospective evaluation on 117 laboratory and engineered sequences

Tier 1 is point classification against the 50 °C threshold. Tier 2 is whether the predicted 95 %
interval is consistent with the reference class; it is a consistency check on the stated
uncertainty, not an accuracy measure, and it improves monotonically with interval width, so the
mean half-width is given alongside. Intervals use the variance scale fitted in §3.5
(*c* = 1.56, μ ± 1.96·*c*·σ).

| Cohort | N | Reference type | Point agreement (°C) | Tier 1 ↑ | Tier 2 ↑ | Mean 95 % half-width (°C) |
|:---|:---:|:---|:---:|:---:|:---:|:---:|
| Codon-optimised 5OCR series | 5 | Threshold label | not available | **100.0 %** (5/5) | **100.0 %** (5/5) | 7.3 |
| High-activity carrageenases | 2 | Measured *T*<sub>opt</sub> | **2.29** vs *T*<sub>opt</sub> | **100.0 %** (2/2) | **100.0 %** (2/2) | 6.0 |
| Thermolabile lipases | 45 | Threshold label | not available | 51.1 % (23/45) | 77.8 % (35/45) | 5.9 |
| Thermostable lipases | 52 | Threshold label | not available | 38.5 % (20/52) | 80.8 % (42/52) | 6.5 |
| **All scored** | **104** | — | — | **48.1 %** | **80.8 %** | **6.2** |
| Carrageenase test suite | 13 | Prospective, no reference | — | — | — | — |

No measured melting temperature exists for the 5OCR series or either lipase cohort, so no point
error is reported for them. The carrageenase figure is agreement with the measured activity
optimum, a related but distinct quantity that is typically lower than the melting temperature.

Under the wider out-of-distribution variance scale (*c* = 3.45, mean half-width 13.8 °C) Tier 2
rises to 93.3 % across all scored sequences and 100 % for the thermostable lipases. That gain is
bought with width and is reported only to show the sensitivity.
