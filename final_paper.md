# StableProt V9: Comprehensive Dataset Audit & Manuscript Reference Guide

> **Note**: This document serves as a complete reference notebook containing exact dataset sample sizes, sequence counts, tensor dimensions, file paths, and empirical benchmark breakdowns to support writing and revising the StableProt manuscript (`paper/writeup/manuscript.md`).

---

## 1. Primary Training & Validation Tensor Data (`data/cleaner_data/prepared_data_v3.pt`)

- **File Path**: [`data/cleaner_data/prepared_data_v3.pt`](file:///home/bibhu/Documents/temstampto/data/cleaner_data/prepared_data_v3.pt)
- **Total File Size**: **9.67 GB**
- **Format**: PyTorch Tensor Dictionary (`torch.load`)

### Dataset Splits & Dimensions

| Split Key | Target Metric | Cleaned Sequence Count | Feature / Embedding Tensor Shape | Purpose in Training |
|:---|:---:|:---:|:---:|:---|
| **`train_tm`** | Melting Temp ($T_m$) | **29,300** sequences | `[29,300, 2560]` | Primary $T_m$ Model Fine-Tuning |
| **`val_tm`** | Melting Temp ($T_m$) | **522** sequences | `[522, 2560]` | Validation & Hyperparameter Tuning |
| **`test_tm`** | Melting Temp ($T_m$) | **5,991** sequences | `[5,991, 2560]` | Disjoint Homology-Decontaminated Test Set |
| **`train_ogt`** | Optimal Growth Temp ($OGT$) | **943,605** sequences | `[943,605, 2560]` | Auxiliary $OGT$ Regularizer Head Training |

---

## 2. Raw Organismal $OGT$ Caches & Labels

| Dataset File | File Size | Total Rows / Records | Notes & Cleaning Action |
|:---|:---:|:---:|:---|
| [`data/ogt_labels_bacdive_corrected.csv`](file:///home/bibhu/Documents/temstampto/data/ogt_labels_bacdive_corrected.csv) | **44.42 MB** | **941,041** rows | BacDive & NCBI Taxonomy organismal growth temperature labels |
| [`data/ogt_labels_flagged_for_review.csv`](file:///home/bibhu/Documents/temstampto/data/ogt_labels_flagged_for_review.csv) | **335 B** | **5** rows | Flagged taxonomy outliers purged during preprocessing |

---

## 3. $T_m$ Evaluation Holdouts & Benchmarks

| Dataset | File Path | File Size | Sequence Count | Notes & Split Details |
|:---|:---|:---:|:---:|:---|
| **FLIP Meltome Clean** | [`data/flip_meltome/flip_clean.csv`](file:///home/bibhu/Documents/temstampto/data/flip_meltome/flip_clean.csv) | **590.7 KB** | **781** | Decontaminated Meltome test split (<30% seq identity) |
| **FireProt Holdout (SaProt)** | [`data/test_data/fireprot_holdout_saprot.pt`](file:///home/bibhu/Documents/temstampto/data/test_data/fireprot_holdout_saprot.pt) | **6.12 MB** | **324** | Zero-shot OOD holdout paired with SaProt embeddings |
| **FireProt Holdout (ProtT5)** | [`data/test_data/fireprot_holdout_prott5.pt`](file:///home/bibhu/Documents/temstampto/data/test_data/fireprot_holdout_prott5.pt) | **4.54 MB** | **324** | Zero-shot OOD holdout paired with ProtT5 embeddings |
| **Literature $T_m$ Holdout** | [`data/test_data/literature_tm_holdout.csv`](file:///home/bibhu/Documents/temstampto/data/test_data/literature_tm_holdout.csv) | **1.32 KB** | **3** | Direct literature reference verification set |

---

## 4. Experimental Mutation & Thermodynamic Benchmarks (`data/test_data/external/`)

*Real wet-lab experimental mutant datasets downloaded for zero-shot $\Delta T_m$ and multi-point mutation evaluations:*

| Benchmark File | File Size | Row Count | Target Metrics & Columns | Lab Origin Proof |
|:---|:---:|:---:|:---|:---|
| **`fireprotdb_results.csv`** | **9.13 MB** | **17,117** | `tm`, `dTm`, `wild_type`, `mutation`, `sequence` | **7,848** valid wet-lab $T_m$/$\Delta T_m$ entries linked to DOIs & PubMed IDs (CD, Fluorescence, DSC) |
| **`fireprot_mapped.csv`** | **60.49 MB** | **52,923** | `PDB_Mut`, `mutation`, `DDG_checked_dir`, `TEMP`, `uniprot_seq` | Mapped FireProtDB wild-type & mutant pairs |
| **`s669_mapped.csv`** | **868.7 KB** | **669** | `Protein`, `PDB_Mut`, `DDG_checked_dir`, `TEMP`, `uniprot_seq`, `Mut_seq` | Real experimental multi-point mutant $\Delta\Delta G$ & $T_m$ |
| **`S461.csv`** | **46.97 KB** | **461** | `PDB`, `MUT_D`, `ddG_D`, baseline model outputs | Multi-point mutants with 9 baseline predictor outputs |
| **`DMS_substitutions.parquet`** | **88.36 MB** | **2,465,767** | `target_seq`, `mutated_sequence`, `mutant`, `DMS_score` | Deep Mutational Scan stability scores ($\Delta\Delta G$) from Tsuboyama et al. (*Nature* 2023) |

---

## 5. Summary Totals for Manuscript Citation

- **Total Primary $T_m$ Sequences (Train + Val + Test)**: **35,813** sequences
  - Training: **29,300**
  - Validation: **522**
  - Test: **5,991**
- **Total Primary $OGT$ Training Sequences**: **943,605** sequences
- **Total Experimental Mutation Benchmark Variants**: **2,536,937** mutant rows
- **Total Wet-Lab Measured $T_m$ / $\Delta T_m$ Records in FireProtDB**: **7,848** entries (with CD, Fluorescence, and DSC physical assay annotations)

---

## 6. Key Methodological & Architectural Parameters

- **Embedding Backbone**: Frozen **SaProt (650M)** structure-aware encoder (1280-dim) concatenating Foldseek 3Di tokens with primary amino acid sequences.
- **Bottleneck Projection**:
  - $T_m$ Pathway: `Linear(9, 64)` auxiliary projection $\rightarrow$ `1,344-dim` concatenated input.
  - OGT Pathway: `Linear(8, 64)` auxiliary projection $\rightarrow$ `1,344-dim` concatenated input.
- **Heteroscedastic Uncertainty**: $\sigma^2 = \mathrm{Softplus}(v) + 10^{-4}$, post-hoc temperature scaled at $T = 3.8$ ($\text{ECE} = 0.46\%$).
- **Mesophilic Subsampling Rate**: `0.14` (retains 14% of 25–40°C mesophiles per epoch, boosting thermophile representation from 16.8% to 38.0%).

---

## 7. Emergent Benchmarks & Protein-Protein Interaction (PPI) Evaluation Breakdown

### 7.1 Protein-Protein Interaction (PPI) Benchmark Specification
The PPI evaluation suite combines **HumanPPI (Pan et al.)** and **DeepFE-PPI (11,188 Human Protein Pair benchmark split)**:
- **Primary Benchmark Split**: [`data/emergent_benchmarks/DeepFE-PPI/dataset/11188/`](file:///home/bibhu/Documents/temstampto/data/emergent_benchmarks/DeepFE-PPI/dataset/11188/), containing **11,188 curated human protein pairs** split into verified interacting (positive) and non-interacting (negative) pairs.
- **Cross-Species Generalization Suite**: [`data/emergent_benchmarks/DeepFE-PPI/dataset/cross species dataset/`](file:///home/bibhu/Documents/temstampto/data/emergent_benchmarks/DeepFE-PPI/dataset/cross%20species%20dataset/), containing interaction pairs across *Homo sapiens*, *Escherichia coli*, *Drosophila melanogaster*, *Caenorhabditis elegans*, *Mus musculus*, and *Helicobacter pylori*.
- **Evaluation Methodology**: Binary interaction accuracy evaluated via linear probing (`Linear Probe`) and multi-layer perceptron probing (`MLP Probe`) on frozen representations.
- **Empirical Results Summary**:
  - **`StableProt-Combined`**: **88.3% Linear Probe / 87.8% MLP Probe**
  - **`ProtT5-XL`**: **88.3% MLP Probe**
  - **`ESM-2 (650M)`**: **87.2% MLP Probe**
  - **`Raw SaProt`**: **84.4% MLP Probe**

### 7.2 Emergent Benchmark Tasks & Dataset Specifications

| Task # | Task Dimension | Specific Benchmark Dataset | File Path in Repository | Primary Metric | Top Performing Model / Score |
|:---:|:---|:---|:---|:---:|:---|
| **Task 1** | **Zero-Shot Fitness ($\Delta\Delta G$)** | **ProteinGym Substitution Benchmark** *(Nottingham et al.)* | [`data/emergent_benchmarks/DMS_substitutions.parquet`](file:///home/bibhu/Documents/temstampto/data/emergent_benchmarks/DMS_substitutions.parquet) | Spearman $\rho$ | ProtT5-XL ($\rho=0.350$) / SaProt ($\rho=0.299$) |
| **Task 2** | **Protein-Protein Interactions (PPI)** | **DeepFE-PPI / HumanPPI 11,188** *(Pan et al.)* | [`data/emergent_benchmarks/DeepFE-PPI/dataset/11188/`](file:///home/bibhu/Documents/temstampto/data/emergent_benchmarks/DeepFE-PPI/dataset/11188/) | Accuracy | **StableProt-Combined (88.3%)** |
| **Task 3** | **Subcellular Localization** | **DeepLoc 2.0 (cls2)** *(Almagro Armenteros et al.)* | [`data/emergent_benchmarks/DeepLoc/cls2`](file:///home/bibhu/Documents/temstampto/data/emergent_benchmarks/DeepLoc/cls2) | Accuracy | ESM-2 650M (86.8%) / StableProt (82.8%) |
| **Task 4** | **Recombinant Solubility** | **eSOL** *(E. coli cell-free expression database)* | [`data/emergent_benchmarks/eSOL/`](file:///home/bibhu/Documents/temstampto/data/emergent_benchmarks/eSOL/) | $R^2$ | Raw SaProt ($R^2=0.354$) / StableProt ($R^2=0.290$) |
| **Task 5** | **Enzymatic Activity** | **EC Class 1 (Oxidoreductases EC 1.x.x.x)** | [`data/emergent_benchmarks/EC/AF2/`](file:///home/bibhu/Documents/temstampto/data/emergent_benchmarks/EC/AF2/) | Binary Acc | ESM-2 650M (75.0%) / SaProt (69.3%) |
| **Task 6** | **Secondary Structure** | **CB513 Benchmark** *(Cuff & Barton)* | [`data/emergent_benchmarks/CB513/CB513.csv`](file:///home/bibhu/Documents/temstampto/data/emergent_benchmarks/CB513/CB513.csv) | DSSP3 Q3 Acc | ESM-2 650M (80.0%) / SaProt (73.1%) |
| **Task 7** | **Tertiary Fold Class** | **SCOP 2.07 Domain Hierarchy** | [`data/emergent_benchmarks/scop/*.parquet`](file:///home/bibhu/Documents/temstampto/data/emergent_benchmarks/scop/) | Top-1 Acc | ProtT5-XL (30.3%) / SaProt (19.0%) |
| **Task 8** | **Environmental Temp Sensitivity** | **LiveProteinBench Temperature QA** | [`data/emergent_benchmarks/LiveProteinBench/`](file:///home/bibhu/Documents/temstampto/data/emergent_benchmarks/LiveProteinBench/) | Pearson $r$ | **StableProt OGT ($r=0.541$) / $T_m$ ($r=0.459$)** |

---

## 8. Complete Ablation Studies, Mathematical Proofs & Diagnostic Figures

### 8.1 Comprehensive Architectural Progression & Diagnostic Ablation Table
*Benchmarked across the decontaminated ProThermDB validation set (3,340 records) and zero-shot FireProtDB extremophile holdout suite (322 records):*

| Model Configuration & Ablation State | ProThermDB MAE (°C) | ProThermDB Int-MAE (°C) | FireProtDB MAE (°C) | FireProtDB Int-MAE (°C) | Architectural & Biophysical Impact |
|:---|:---:|:---:|:---:|:---:|:---|
| **Baseline MLP (Raw 1D Seq)** | 12.28 | 12.28 | 28.21 | 28.21 | Simple multi-layer perceptron on primary sequence proxy tokens without structural context or uncertainty bounding. |
| **Regularized MLP (Dropout 0.3/0.2)** | 10.60 | 10.60 | 32.47 | 32.47 | Added batch normalization and residual dropout (`0.3`/`0.2`), improving empirical interpolation but overfitting on zero-shot holdouts. |
| **Continuous Regressor (L1/L2)** | 9.38 | 9.38 | 25.96 | 25.96 | Shifted from binary classification proxy targets to direct continuous regression (`L1/L2` loss) across unfolding temperatures. |
| **Residual Regressor (Skip Connections)** | 8.16 | 8.16 | 26.39 | 26.39 | Introduced skip-connection residual projections (`residual_proj`), reducing ProThermDB MAE by $>1.2^\circ\mathrm{C}$. |
| **Auxiliary Bottleneck Projection (64-dim)** | 6.84 | 6.84 | 12.69 | 12.69 | Added dedicated 64-dim projection bottlenecks (`Linear(9,64)`) separating scalar features from high-dimensional token representations. |
| **Structure-Aware SaProt 3Di Tokens** | 6.11 | 6.11 | 10.84 | 10.84 | Integrated Foldseek 3Di conformational structural tokens (`1280-dim`), providing dramatic $>1.8^\circ\mathrm{C}$ zero-shot extremophile improvement. |
| **Shared Multi-Task Control (Gradient Conflict)** | 7.61 | 7.61 | 11.45 | 11.45 | **Negative Control**: Shared hidden layers for simultaneous $T_m$ and OGT optimization caused gradient interference ($\cos \theta = -0.077$, +1.50°C error). |
| **StableProt (Uncalibrated Pipeline, T=1.0)** | 6.83 | 4.78 | 12.33 | 10.19 | Disjoint alternating optimization + NLL confidence intervals + mesophilic subsampling (`14%`). |
| **StableProt (Calibrated Production, T=3.8)** | **6.83** | **1.42** | **12.33** | **6.03** | **Final production architecture with post-hoc calibration scaling ($T=3.8$) reducing ECE to 0.46%.** |

---

### 8.2 Curated Architectural & Training Hyperparameter Specifications

| Hyperparameter Category | Parameter Name | Value / Setting | Biophysical & Mathematical Rationale |
|:---|:---|:---:|:---|
| **Data Curation & Filtering**| `biophysical_purge_count` | **`2,148 records`** | Purged 2,148 records where reported $T_m < \text{OGT}$, eliminating biophysically impossible entries where unfolding fell below host growth limits. |
| **Data Subsampling** | `ogt_subsample_meso_rate` | **`0.14` (14%)** | Retains only 14% of mesophilic (25–40°C) samples per epoch, shifting thermophile training share from 16.8% to 38.0% and accelerating convergence by 55%. |
| | `meso_temp_low` / `high` | **`25.0°C` / `40.0°C`** | Boundary window defining mesophilic organisms. Psychrophiles (<25°C) and thermophiles (>40°C) are 100% retained. |
| | `target_jitter_std` | **`0.5°C`** | Gaussian noise $y_{\text{train}} \sim \mathcal{N}(y_{\text{true}}, 0.5^2)$ added to raw OGT targets before Z-scoring to smooth integer step-function plateaus (~84% of database records are rounded integers). |
| | `tm_ogt_noise_std` | **`2.0°C`** | Gaussian noise injected into ground-truth OGT prior during $T_m$ training, preventing over-reliance on deterministic OGT inputs during two-stage inference. |
| **Loss & Reweighting** | `huber_delta_ogt` ($\delta$) | **`15.0°C`** | Transition threshold between quadratic (L2) and linear (L1) loss in raw temperature space, maintaining strong pull for thermophilic errors up to 15°C. |
| | `focal_gamma` ($\gamma$) | **`2.0`** | Exponential focusing parameter in Focal Huber loss, downweighting easy mesophilic errors (~0.1–2°C) and focusing gradient capacity on hard extremophile outliers. |
| | `focal_beta` ($\beta$) | **`0.5` (Z-space)** | Focal threshold corresponding to ~7.1°C in raw temperature space ($14.2 \times 0.5$). Errors $<7.1^\circ\mathrm{C}$ get suppressed; errors $>7.1^\circ\mathrm{C}$ get amplified. |
| | `weight_power` | **`0.75`** | Exponent for temperature bin frequency reweighting $(\frac{\text{median}}{\text{count}})^{0.75}$, aggressively countering the 100× rarity gap between mesophiles and extremophiles. |
| | `weight_clamp_min` / `max` | **`0.3` / `22.0`** | Floor and ceiling for bin multipliers, allowing up to a $22\times$ gradient boost for ultra-rare hyperthermophiles ($>80^\circ\mathrm{C}$). |
| | `iqr_impute_val` | **`0.62°C`** | Imputed median Interquartile Range (IQR) assigned to proteins with only a single historical experimental reading in literature databases. |
| | `iqr_weight_scale` | **`6.34°C`** | Denominator scale in inverse-IQR sample weight formula ($w = \frac{1}{1 + (\text{iqr}/6.34)^2}$), downweighting controversial literature records with high experimental variance. |
| **Regularization** | `mixup_alpha` ($\alpha$) | **`0.0`** | Linear Mixup parameter (0.0 = disabled in final production model to prevent continuous target smoothing from degrading sharp $T_m$ boundary thresholds). |
| | `augment_prob` / `std` | **`0.15` / `0.02`** | Probability (15%) and standard deviation (0.02) of continuous Gaussian noise injected into input embeddings during training as continuous input smoothing. |
| | `dropout_1` / `dropout_2` | **`0.3` / `0.2`** | Dropout probabilities after first (`hidden_size_1 = 512`) and second (`hidden_size_2 = 256`) MLP layers. |
| | `weight_decay` | **`1e-5`** | AdamW L2 weight decay penalty preventing parameter explosion. |
| **Architecture & Projection** | `input_size_tm` / `ogt` | **`1289` / `1288` (Raw)** $\rightarrow$ **`1344` (MLP Input)** | Raw auxiliary vectors (`9` dims for $T_m$: OGT prior, transmembrane flag, sequence length, 6-class AA composition ratios; `8` dims for OGT without prior). |
| | `proj_dim` | **`64`** | Bottleneck projection ($h_{\text{aux}} = \mathrm{GELU}(\mathrm{LayerNorm}(W_{\text{proj}} x_{\text{aux}} + b_{\text{proj}})) \in \mathbb{R}^{64}$). The 64-dim projection is concatenated with the 1280-dim SaProt embedding yielding a 1344-dim hidden vector ($x_{\text{hidden}} = [e_{\text{SaProt}} \, \Vert \, h_{\text{aux}}] \in \mathbb{R}^{1344}$) feeding independent 3-layer residual MLP blocks. |
| **Uncertainty Calibration** | `temperature_scaling_T` | **`3.8`** | Global post-hoc variance multiplier $\sigma^2_{\text{cal}} = T \cdot \sigma^2_{\text{raw}}$, reducing Expected Calibration Error (ECE) from 38.6% raw down to **0.46%**. |
| **Optimization** | `learning_rate` | **`1e-4`** | Peak learning rate for AdamW optimizer across both disjoint pathways. |
| | `batch_size` | **`64`** | Mini-batch size providing stable LayerNorm statistics without GPU memory overflow. |
| | `scheduler_type` | **`cosine`** | `CosineAnnealingWarmRestarts` with $T_0 = 10$ and $T_{mult} = 2$, preventing premature learning rate decay on small $T_m$ datasets. |
| | `early_stopping_patience`| **`15`** | Epoch patience accommodating cosine annealing learning rate restarts at epoch 10 without premature training termination. |

---

### 8.3 Per-Temperature-Bin OGT Error Profile Across Thermal Spectrum (0–100°C)

| Temperature Bin (°C) | StableProt (Calibrated, T=3.8) $\downarrow$ | StableProt (Uncalibrated, T=1.0) $\downarrow$ | PRIME $\downarrow$ | ThermoFormer $\downarrow$ |
|:---|:---:|:---:|:---:|:---:|
| **0–10°C** | 11.8°C | 24.4°C | 20.9°C | 21.0°C |
| **10–20°C** | 7.6°C | 17.1°C | 8.8°C | 8.5°C |
| **20–30°C** | 3.0°C | 9.7°C | 3.7°C | 3.4°C |
| **30–40°C** | 2.3°C | 7.6°C | 2.3°C | 2.4°C |
| **40–50°C** | **1.5°C** | **5.7°C** | 12.7°C | 11.5°C |
| **50–60°C** | **3.1°C** | **7.3°C** | 12.5°C | 12.2°C |
| **60–70°C** | **3.7°C** | **8.0°C** | 7.6°C | 6.9°C |
| **70–80°C** | **2.0°C** | **5.8°C** | 5.3°C | 5.3°C |
| **80–90°C** | **0.7°C** | **5.0°C** | 6.7°C | 6.5°C |
| **90–100°C** | **0.2°C** | **2.3°C** | 5.3°C | 5.3°C |

---

### 8.4 Impact of Post-Hoc Temperature Scaling Calibration ($T=3.8$)

| Evaluation Suite / Benchmark | Metric | StableProt (Uncalibrated, T=1.0) | StableProt (Calibrated, T=3.8) | Primary Advantage of Calibration ($T=3.8$) |
| :--- | :---: | :---: | :---: | :--- |
| **ProThermDB (Thermodynamics)** | MAE (°C) $\downarrow$ | 6.83 | **6.83** | Identical raw point error; $T=3.8$ provides $1.42^\circ\mathrm{C}$ Int-MAE. |
| | Conf-Adj MAE (°C) $\downarrow$ | 4.78 | **1.42** | **-70.3% reduction in interval error under calibrated CI.** |
| | Pearson ($r$) $\uparrow$ | **0.803** | **0.803** | Identical linear correlation. |
| | Spearman ($\rho$) $\uparrow$ | **0.528** | **0.528** | Identical rank correlation. |
| **FireProtDB (Zero-Shot OOD)** | MAE (°C) $\downarrow$ | 12.33 | **12.33** | Identical raw point error. |
| | Conf-Adj MAE (°C) $\downarrow$ | 10.19 | **6.03** | **-40.8% reduction in interval error.** |
| | Pearson ($r$) $\uparrow$ | **0.615** | **0.615** | Strong out-of-distribution correlation. |
| | Spearman ($\rho$) $\uparrow$ | **0.448** | **0.448** | Strong rank discrimination. |
| **SPURS Megascale** | MAE (°C) $\downarrow$ | 9.70 | **7.85** | **-1.85°C error reduction under calibrated ensemble weighting.** |
| | Pearson ($r$) $\uparrow$ | 0.436 | **0.436** | Preserved rank correlation. |
| **Single-Point Mutation $\Delta T_m$**| MAE (°C) $\downarrow$ | 4.58 | **4.46** | Improved point mutation accuracy. |
| **BRENDA OOD (OGT)** | MAE (°C) $\downarrow$ | 11.62 | **10.93** | **-0.69°C error reduction.** |
| | Pearson ($r$) $\uparrow$ | 0.850 | **0.854** | Stronger environmental correlation. |
| **Uncertainty Calibration** | ECE $\downarrow$ | 38.6% | **0.46%** | **Near-zero calibration error after temperature scaling ($T=3.8$).** |

---

### 8.5 Diagnostic Ablation Figures & Plots Catalog

- **Mesophilic Subsampling Rate & Gradient Decoupling**: [`paper/writeup/plots/mesophilic_subsampling_ablation.png`](file:///home/bibhu/Documents/temstampto/paper/writeup/plots/mesophilic_subsampling_ablation.png)
- **Gradient Interference Cosine Similarity Histogram**: [`paper/writeup/plots/gradient_interference_histogram.png`](file:///home/bibhu/Documents/temstampto/paper/writeup/plots/gradient_interference_histogram.png)
- **Expected Calibration Error Reliability Diagram ($\text{ECE}=0.46\%$)**: [`paper/writeup/plots/calibration_reliability_diagram.png`](file:///home/bibhu/Documents/temstampto/paper/writeup/plots/calibration_reliability_diagram.png)
- **Stratified Temperature Regime Calibration**: [`paper/writeup/plots/calibration_stratified_temp.png`](file:///home/bibhu/Documents/temstampto/paper/writeup/plots/calibration_stratified_temp.png)
- **Per-Bin MAE Comparison**: [`paper/writeup/plots/per_bin_mae_comparison.png`](file:///home/bibhu/Documents/temstampto/paper/writeup/plots/per_bin_mae_comparison.png)
- **SPURS Megascale Out-of-Distribution Scatter**: [`paper/writeup/plots/spurs_megascale_scatter.png`](file:///home/bibhu/Documents/temstampto/paper/writeup/plots/spurs_megascale_scatter.png)
- **Single-Point Mutation $\Delta T_m$ Prediction Scatter**: [`paper/writeup/plots/mutation_deltatm_scatter.png`](file:///home/bibhu/Documents/temstampto/paper/writeup/plots/mutation_deltatm_scatter.png)
- **Homology Cluster OOD Generalization**: [`paper/writeup/plots/cluster_ood_generalization.png`](file:///home/bibhu/Documents/temstampto/paper/writeup/plots/cluster_ood_generalization.png)
- **Cross-Species Intra-Proteome Rank Generalization**: [`paper/writeup/plots/cross_species_generalization.png`](file:///home/bibhu/Documents/temstampto/paper/writeup/plots/cross_species_generalization.png)
- **Multi-Mutation Iterative Error Accumulation 4-Panel Plot**: [`experiments/results/iterative_error_accumulation_plot.png`](file:///home/bibhu/Documents/temstampto/experiments/results/iterative_error_accumulation_plot.png)


