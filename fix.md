# Project Audit: Placeholders, Redundancies, and Paper Risks

This document tracks identified placeholders, redundant scripts, and critical issues that must be addressed to ensure the scientific integrity of the StableProt V2 publication.

## 1. Placeholder Detection

### [CRITICAL] Synthetic Literature Baselines
The following scripts use synthetic noise models to "simulate" external baselines (ESMStabP, TemBERTure) instead of executing real inference on the holdout set:
- [evaluate_fireprot_generalization.py](file:///home/bibhu/Documents/temstampto/experiments/analysis/evaluate_fireprot_generalization.py) (lines 37-48)
- [compare_all_prothermdb.py](file:///home/bibhu/Documents/temstampto/experiments/analysis/compare_all_prothermdb.py) (lines 210-220)

> [!WARNING]
> Reporting these as direct comparisons in a paper is scientifically misleading. True comparisons require running the literature models on the exact same holdout sequences.

### [CRITICAL] Homology Filtering Methodology
The [curate_fireprot_holdout.py](file:///home/bibhu/Documents/temstampto/experiments/data_processing/curate_fireprot_holdout.py) script uses `difflib.SequenceMatcher` to calculate "sequence identity" ratios. 
- **Issue**: `SequenceMatcher` is a general-purpose string matching heuristic, not a biological alignment tool. It does not account for gaps or substitution matrices (e.g., BLOSUM).
- **Risk**: Reviewers may reject the "Out-of-Distribution" claim if it is not backed by standard tools like `CD-HIT`, `MMSeqs2`, or `BLASTp` with a formal identity threshold.

### [CRITICAL] Ad-Hoc / Dummy Mapping Logic
- [add_ogt_to_tm_datasets.py](file:///home/bibhu/Documents/temstampto/experiments/data_processing/add_ogt_to_tm_datasets.py): Contains explicit "placeholder logic based on typical TSV mapping" and "dummy logic" for Organism-to-OGT mapping. 
- **Risk**: If the organism-to-OGT mapping uses dummy logic, the entire Multi-Task loss weighting is fundamentally flawed because the targets are synthetic or arbitrary.

### [CRITICAL] Global Clustering Audit (CD-HIT 30)
It is unclear if a global CD-HIT run was performed on the *union* of all datasets (Meltome + ProThermDB + FireProt).
- **Current State**: FireProt sequences are screened against training sequences individually. 
- **Requirement**: Standard protocol for OOD validation is to cluster all sequences together at 30% identity and ensure no cluster contains both a training record and a test record.
- **Status**: Not currently implemented for the FireProt holdout.

---

## 2. Future Dataset Requirements

### [REQUIRED] Literature-Sourced Tm Dataset
To properly benchmark and validate generalization without relying on synthetic or circular data, **you must curate a new dataset of novel proteins with experimentally verified $T_m$ values from recent literature.**
- **Why**: ProThermDB and Meltome are heavily saturated in current models. A truly independent test set sourced manually from literature guarantees zero leakage into any baseline model (ESMStabP, TemBERTure).

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

## 3. Benchmarking Improvements

### Synthetic Baselines Removed
- **Status**: Removed synthetic "placeholder" literature baselines (TemBERTure, ESMStabP) from `evaluate_fireprot_generalization.py` and `compare_all_prothermdb.py`.
- **Why**: Synthesizing predictions using mathematical noise is scientifically invalid. True benchmarking requires running the actual baseline models on our exact holdout sets to get **actual numbers**, rather than mimicking reported summary statistics from their papers.

### Per-Sequence Results (CSV/TSV)
- **Status**: Added CSV generation to `evaluate_fireprot_generalization.py`.
- **Output**: [fireprot_benchmarking_results.csv](file:///home/bibhu/Documents/temstampto/experiments/analysis/fireprot_benchmarking_results.csv) now contains per-sequence predictions for all model iterations (V0-V6).

---

## 4. Operational Notes

> [!IMPORTANT]
> **No destructive tasks** (file deletion, large-scale refactoring) will be performed without explicit user permission. The cleanup of legacy scripts should be handled by moving them to an archive directory rather than deletion.
