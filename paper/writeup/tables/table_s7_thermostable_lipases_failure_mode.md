# Supplementary Table S7: Failure mode on the thermostable lipases (N = 52, reference class ≥ 50 °C)

Every sequence in this cohort is labelled thermostable, so the first two rows partition the
cohort and the third describes where the incorrect predictions land. Intervals are
μ ± 1.96·*c*·σ with *c* = 1.56.

| Outcome | Count | % of cohort |
|:---|:---:|:---:|
| Point estimate ≥ 50 °C — classified correctly | 20 | 38.5 % |
| Point estimate < 50 °C — classified incorrectly | 32 | 61.5 % |
| *of which* predicted between 42 and 49 °C | 28 | 53.8 % |

| Interval outcome | Count | % of cohort |
|:---|:---:|:---:|
| 95 % interval reaches ≥ 50 °C — consistent with the label | 42 | 80.8 % |
| 95 % interval falls entirely below 50 °C — inconsistent | 10 | 19.2 % |

The point estimates cluster immediately below the threshold rather than scattering, which is the
signature of under-prediction on thermostable proteins reported in §3.1 and §3.5 rather than of
random error. These sequences carry no host growth-temperature annotation, so the auxiliary OGT
prior takes the corpus mean; the dependency is quantified in §4. The interval covers the
threshold for 42 of the 52 sequences, that is, in most of the misclassified cases the model
signals that it cannot resolve which side of 50 °C the protein lies on. Under the wider
out-of-distribution variance scale (*c* = 3.45) the interval is consistent for all 52, which
illustrates that this measure is bought with width and is not an accuracy statement.
