# Supplementary Table S7: Failure mode on the thermostable lipases (N = 37 unique consensus sequences, reference class ≥ 50 °C)

Every sequence in this clean cohort is uniquely evaluated under single-label biological consensus. The first two rows partition the cohort and the third describes where the incorrect predictions land. Intervals are μ ± 1.96·*c*·σ with *c* = 1.56.

| Outcome | Count | % of cohort |
|:---|:---:|:---:|
| Point estimate ≥ 50 °C — classified correctly | 16 | 43.2 % |
| Point estimate < 50 °C — classified incorrectly | 21 | 56.8 % |
| *of which* predicted between 42 and 49 °C | 17 | 45.9 % |

| Interval outcome | Count | % of cohort |
|:---|:---:|:---:|
| 95 % interval reaches ≥ 50 °C — consistent with the label | 29 | 78.4 % |
| 95 % interval falls entirely below 50 °C — inconsistent | 8 | 21.6 % |

The point estimates cluster immediately below the threshold (17 of 21 misclassified cases land between 42 and 49 °C) rather than scattering randomly across the temperature spectrum, which is the signature of the global mesophilic prior pull reported in §3.1 and §3.5. These sequences carry no host growth-temperature annotation, so the auxiliary OGT prior takes the corpus mean. The calibrated interval covers the threshold for 29 of the 37 sequences (78.4%), indicating that for the majority of misclassified cases, the model signals appropriate boundary uncertainty rather than high-confidence false negatives. Under the wider out-of-distribution variance scale (*c* = 3.45), the interval covers ≥ 50 °C for all 37 sequences (100%).

> **Biological Consensus Deduplication Note**: Initial multi-strain genome assemblies contained 6 housekeeping lipase homologues that were sequence-identical across both thermophilic and mesophilic strain isolate collections. Under biological consensus deduplication, truly mesophilic housekeeping enzymes (spore germination lipase `LipC`, phospholipase `YtpA`, and triacylglycerol lipases) were assigned to the thermolabile cohort ($N = 32$), while `thermostable monoacylglycerol lipase` and `Lipase_1` were assigned to the thermostable cohort ($N = 37$). This yields a fully non-redundant benchmark of $N = 69$ unique lipases ($N = 87$ prospective sequences overall) with zero cross-file duplicates.
