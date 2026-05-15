# Project Audit: Placeholders, Redundancies, and Paper Risks

This document tracks identified placeholders, redundant scripts, and critical issues that must be addressed to ensure the scientific integrity of the StableProt V2 publication.

## 1. Placeholder Detection

### [CRITICAL] Synthetic Literature Baselines
The following scripts use synthetic noise models to "simulate" external baselines (ESMStabP, TemBERTure) instead of executing real inference on the holdout set:
- [evaluate_fireprot_generalization.py](file:///home/bibhu/Documents/temstampto/experiments/analysis/evaluate_fireprot_generalization.py) (lines 37-48)
- [compare_all_prothermdb.py](file:///home/bibhu/Documents/temstampto/experiments/analysis/compare_all_prothermdb.py) (lines 210-220)

> [!WARNING]
> Reporting these as direct comparisons in a paper is scientifically misleading. True comparisons require running the literature models on the exact same holdout sequences.

### [CRITICAL] Misleading Metrics in `reseults.md`
The [reseults.md](file:///home/bibhu/Documents/temstampto/reseults.md) file reports a V6 MAE of **5.82°C** for the FireProt OOD set.
- **Actual Metric**: The real evaluated MAE on the <30% identity FireProt holdout is **16.52°C**.
- **Source of Error**: The 5.82 value likely comes from a ProThermDB (in-distribution) test set and is being incorrectly presented as an OOD result.

### Minor Placeholders
- [devlog.md](file:///home/bibhu/Documents/temstampto/devlog.md): References placeholder labels `999` for unlabeled sequences.
- [add_ogt_to_tm_datasets.py](file:///home/bibhu/Documents/temstampto/experiments/data_processing/add_ogt_to_tm_datasets.py): Contains placeholder mapping logic for OGT-Tm alignment.

---

## 2. Paper Risks & Scientific Issues

### Generalization Floor (PCC 0.25)
The V6 ESM-2 Multi-Head model achieves a Pearson Correlation Coefficient (PCC) of only **0.25** on the FireProt OOD set. 
- **Comparison**: The ProtT5 version (V5) generalizes significantly better (PCC 0.45).
- **Risk**: Claiming SOTA generalization for V6 is difficult with a PCC of 0.25. The model appears heavily overfitted to the ProThermDB/Meltome feature manifold.

### Thermodynamic Mean Shift
The model predicts a mean Tm of ~47°C for FireProt targets, while the actual mean is ~62°C. 
- **Issue**: The model is biased towards the training mean (~50°C). It struggles with the thermophilic bias often found in curated database like FireProt.

---

## 3. Operational Notes

> [!IMPORTANT]
> **No destructive tasks** (file deletion, large-scale refactoring) will be performed without explicit user permission. The cleanup of legacy scripts should be handled by moving them to an archive directory rather than deletion.
