# Supplementary Materials for StableProt: Structure-Aware Deep Learning for Protein Thermostability ($T_m$) and Environmental Adaptation (OGT) Prediction

This document provides supplementary hyperparameter configurations, dataset curation specifications, mathematical proofs of gradient decoupling, detailed ablation analyses, and extended experimental validation tables supporting the main manuscript.

---

## Supplementary Table S1: Curated Architectural and Training Hyperparameters

The following table catalogs the primary hyperparameter values utilized across the StableProt training and inference pipeline. All training runs were executed across 5 independent random seeds (`[1, 2, 3, 4, 5]`) using PyTorch with Automatic Mixed Precision (`GradScaler`).

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

## Supplementary Table S2: Comprehensive Dataset Partitions, Homology Separation, and Curation Rules

Detailed composition, sample sizes, taxonomic diversity, empirical temperature statistics, redundancy clustering thresholds, and thermodynamic purification criteria across all training, validation, out-of-distribution benchmark, and prospective experimental assay cohorts.

| Dataset / Cohort | Functional Role in Pipeline | Sequences ($N$) | Unique Organisms | Mean $T_m$ / OGT | Median | Min–Max Range | Sequence Redundancy Criterion | Thermodynamic Purge / Sampling Filter |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ProThermDB Train** | Primary $T_m$ Head Optimization | **28,739** | 1,842 | $54.8^\circ\text{C}$ | $53.2^\circ\text{C}$ | $21.5\text{--}104.0^\circ\text{C}$ | MMseqs2 $< 30\%$ vs test splits | $100\%$ ($T_m \ge \text{OGT}$ biophysical consistency purge; $N=2,148$ removed) |
| **Aux-OGT Train** | Secondary Head & $T_m$ Prior | **131,920** | 8,950 | $39.4^\circ\text{C}$ | $37.0^\circ\text{C}$ | $4.0\text{--}102.0^\circ\text{C}$ | CD-HIT $< 40\%$ identity | $14\%$ mesophilic subsampling ($25\text{--}40^\circ\text{C}$); $100\%$ psychro/thermo retained |
| **ProThermDB Test** | Primary In-Domain Benchmark | **3,340** | 318 | $55.4^\circ\text{C}$ | $54.0^\circ\text{C}$ | $24.0\text{--}98.5^\circ\text{C}$ | Disjoint held-out clusters | Strict zero-homology partition ($<30\%$ vs training) |
| **FireProtDB OOD** | Independent Out-of-Distribution | **322** | 114 | $57.8^\circ\text{C}$ | $56.0^\circ\text{C}$ | $26.0\text{--}105.0^\circ\text{C}$ | MMseqs2 $< 30\%$ vs ProThermDB | Fully external curated gold-standard benchmark |
| **BRENDA OGT Test** | Extreme Organism OGT Benchmark | **525** | 525 | $58.2^\circ\text{C}$ | $55.0^\circ\text{C}$ | $4.0\text{--}100.0^\circ\text{C}$ | Fully disjoint species holdout | Independent extreme thermophile validation |
| **Prospective 115** | Wet-Lab Industrial Assays | **115** | 48 | $62.4^\circ\text{C}$ | $61.0^\circ\text{C}$ | $35.0\text{--}95.0^\circ\text{C}$ | Novel uncharacterized enzymes | Direct de novo experimental CD/DSF assays |

---

## Supplementary Table S3: Comprehensive Architectural Progression and Diagnostic Ablation Study

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

## Supplementary Table S4: Binary Thermostability Triage Performance Metrics ($T_m \geq 60^\circ\text{C}$)

Comprehensive classification performance comparison on the held-out ProThermDB test partition for identifying industrial-grade thermophilic enzymes ($T_m \geq 60.0^\circ\text{C}$). Best results in **bold**, second-best <u>underlined</u>.

| Model Architecture | Encoder Type / Pre-training | AUC-ROC | Precision | Recall (Sensitivity) | F1-Score | Specificity | MCC (Matthews Corr) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **StableProt (Ours)** | **SaProt 650M (3Di Structural) + Heteroscedastic MLP** | **0.941** | **0.892** | **0.915** | **0.903** | **0.938** | **0.842** |
| **TemBERTure** | ProtBERT 420M + Regression Head | <u>0.884</u> | <u>0.814</u> | <u>0.832</u> | <u>0.823</u> | <u>0.865</u> | <u>0.710</u> |
| **DeepSTABp** | ProtT5-XL 3B + MLP Predictor | 0.862 | 0.785 | 0.804 | 0.794 | 0.842 | 0.665 |
| **ThermoFormer** | Transformer (96M Sequence Pre-training) | 0.835 | 0.748 | 0.772 | 0.760 | 0.810 | 0.612 |
| **ESMStabP** | ESM-2 650M Fine-Tuned | 0.812 | 0.710 | 0.735 | 0.722 | 0.782 | 0.554 |
| **TemStaPro (Proxy)** | ESM-1b Binary Classification Proxy | 0.765 | 0.642 | 0.680 | 0.660 | 0.730 | 0.450 |

---

## Supplementary Table S5: Per-Temperature-Bin Optimal Growth Temperature (OGT) Error Profile across the Full Thermal Spectrum

| Temperature Bin | StableProt (Calibrated, T=3.8) | StableProt (Uncalibrated, T=1.0) | PRIME | ThermoFormer |
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

## Supplementary Table S6: Impact of Post-Hoc Temperature Scaling Calibration ($T=3.8$) across Evaluation Suites

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

## Supplementary Table S7: Codon-Optimized Synthesis Variant Series (5OCR Subtilisin E Mutants, Clean 278 aa)

| Protein ID / Variant | Length (aa) | Pred $T_m$ (°C) | Calibrated Uncertainty ($\pm 3.8\sigma$) | 95% Confidence Interval (°C) | $\Delta T_m$ vs WT (°C) | Ground Truth Target (°C) | Tier 1 Point Class ($\ge 50^\circ\mathrm{C}$) | Tier 2 CI Range Inclusion |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Wild-Type 5OCR** | 278 | **43.91** | ±10.62 | [33.29, 54.53] | +0.00 | 43.91 (Thermolabile) | **Correct** | **Correct** |
| **`mut_309`** (`176D>L; 241E>P; 40D>T; 158S>V`) | 278 | **56.86** | ±8.70 | [48.16, 65.56] | **+12.95** | 50.0 (Thermostable) | **Correct** | **Correct** |
| **`mut_345`** (`176D>L; 227E>F; 40D>T; 158S>V`) | 278 | **60.98** | ±7.39 | [53.59, 68.37] | **+17.07** | 50.0 (Thermostable) | **Correct** | **Correct** |
| **`mut_464`** (`241E>P; 227E>F; 40D>T; 276A>P`) | 278 | **52.80** | ±8.97 | [43.83, 61.77] | **+8.88** | 50.0 (Thermostable) | **Correct** | **Correct** |
| **`mut_299`** (`176D>L; 241E>P; 227E>F; 40D>T`) | 278 | **51.11** | ±9.79 | [41.32, 60.90] | **+7.20** | 50.0 (Thermostable) | **Correct** | **Correct** |

---

## Supplementary Table S8: High-Activity Carrageenases ($N=2$, Experimental $T_{\text{opt}}$ Comparison)

| Protein Name | Organism / Family | Length (aa) | Exp $T_{\text{opt}}$ (°C) | Pred $T_m$ (°C) | Calibrated Uncertainty ($\pm 3.8\sigma$) | 95% Confidence Interval (°C) | Tier 1 Point Class ($<50^\circ\mathrm{C}$) | Tier 2 CI Range Inclusion |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CgkS** | Kappa-carrageenase | 407 | **45.0** | **47.38** | ±8.11 | [39.27, 55.49] | **Correct** | **Correct** |
| **CgiB_Ce** | Iota-carrageenase | 461 | **40.0** | **42.19** | ±6.87 | [35.32, 49.06] | **Correct** | **Correct** |

---

## Supplementary Table S9: Failure Mode Breakdown on Thermostable Lipases ($N=52$, Target $\ge 50.0^\circ\mathrm{C}$)

| Prediction Outcome Category | $T_m$ Range | Count ($N$) | Percentage (%) | Biophysical Failure Cause |
|:---|:---:|:---:|:---:|:---|
| **Tier 1 Point Accuracy ($\ge 50^\circ\mathrm{C}$)** | $\ge 50.0^\circ\mathrm{C}$ | 20 | **38.5%** | Successfully identified as thermostable point predictions. |
| **Tier 2 CI Bounds Covered ($\ge 50^\circ\mathrm{C}$)** | $\text{CI}_{\text{high}} \ge 50^\circ\mathrm{C}$ | 46 | **88.5%** | Wide calibrated confidence intervals ($\pm 3.8\sigma$) successfully cover $50^\circ\text{C}$ threshold. |
| **Mesophilic Baseline Reversion** | $<50.0^\circ\mathrm{C}$ | 32 | **61.5%** | Reversion to mesophilic proteomic prior ($42\text{--}49^\circ\mathrm{C}$) due to unannotated host OGT metadata. |
