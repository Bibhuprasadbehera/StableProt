# Table S2: Comprehensive Dataset Partitions, Homology Separation, and Curation Rules

Detailed composition, sample sizes, taxonomic diversity, empirical temperature statistics, redundancy clustering thresholds, and thermodynamic purification criteria across all training, validation, out-of-distribution benchmark, and prospective experimental assay cohorts.

| Dataset / Cohort | Functional Role in Pipeline | Sequences ($N$) | Unique Organisms | Mean $T_m$ / OGT | Median | Min–Max Range | Sequence Redundancy Criterion | Thermodynamic Purge / Sampling Filter |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ProThermDB Train** | Primary $T_m$ Head Optimization | **28,739** | 1,842 | $54.8^\circ\text{C}$ | $53.2^\circ\text{C}$ | $21.5\text{--}104.0^\circ\text{C}$ | MMseqs2 $< 30\%$ vs test splits | $100\%$ ($T_m \ge \text{OGT}$ biophysical consistency purge; $N=2,148$ removed) |
| **Aux-OGT Train** | Secondary Head & $T_m$ Prior | **131,920** | 8,950 | $39.4^\circ\text{C}$ | $37.0^\circ\text{C}$ | $4.0\text{--}102.0^\circ\text{C}$ | CD-HIT $< 40\%$ identity | $14\%$ mesophilic subsampling ($25\text{--}40^\circ\text{C}$); $100\%$ psychro/thermo retained |
| **ProThermDB Test** | Primary In-Domain Benchmark | **3,340** | 318 | $55.4^\circ\text{C}$ | $54.0^\circ\text{C}$ | $24.0\text{--}98.5^\circ\text{C}$ | Disjoint held-out clusters | Strict zero-homology partition ($<30\%$ vs training) |
| **FireProtDB OOD** | Independent Out-of-Distribution | **322** | 114 | $57.8^\circ\text{C}$ | $56.0^\circ\text{C}$ | $26.0\text{--}105.0^\circ\text{C}$ | MMseqs2 $< 30\%$ vs ProThermDB | Fully external curated gold-standard benchmark |
| **BRENDA OGT Test** | Extreme Organism OGT Benchmark | **525** | 525 | $58.2^\circ\text{C}$ | $55.0^\circ\text{C}$ | $4.0\text{--}100.0^\circ\text{C}$ | Fully disjoint species holdout | Independent extreme thermophile validation |
| **Prospective 115** | Wet-Lab Industrial Assays | **115** | 48 | $62.4^\circ\text{C}$ | $61.0^\circ\text{C}$ | $35.0\text{--}95.0^\circ\text{C}$ | Novel uncharacterized enzymes | Direct de novo experimental CD/DSF assays |

### Curation Pipeline Parameters:
1. **Length Boundaries:** $50 \leq L \leq 2,000\,\text{amino acids}$.
2. **Alphabet Sanitation:** Non-canonical symbols (`X`, `B`, `Z`, `J`) discarded.
3. **Measurement Consensus:** Replicate $T_m$ records for identical polypeptides grouped; groups with $T_m\ \text{spread} > 10.0^\circ\text{C}$ purged as experimental outliers; consensus assigned as sample median.
4. **Biophysical Consistency:** Records with $T_m < \text{host OGT}$ purged ($N=2,148$).
5. **Continuous Target Jitter:** $\sigma = 0.5^\circ\text{C}$ Gaussian noise during training forward passes to smooth database integer rounding plateaus (84% of raw OGT values are integers).
