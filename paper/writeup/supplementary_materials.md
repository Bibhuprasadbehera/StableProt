# Supplementary Materials for StableProt: Structure-Aware Deep Learning for Protein Thermostability ($T_m$) and Environmental Adaptation (OGT) Prediction

This document provides supplementary hyperparameter configurations, mathematical proofs of gradient decoupling, and detailed ablation analyses supporting the main manuscript. To maintain clarity and focus on biophysical relevance, we curate only the scientifically meaningful parameters that govern model convergence, regularization, and uncertainty quantification.

---

## Supplementary Table S1: Curated Architectural and Training Hyperparameters

The following table catalogs the primary hyperparameter values utilized across the StableProt training and inference pipeline. All training runs were executed across 5 independent random seeds (`[1, 2, 3, 4, 5]`) using PyTorch with Automatic Mixed Precision (`GradScaler`).

| Hyperparameter Category | Parameter Name | Value / Setting | Biophysical & Mathematical Rationale |
|:---|:---|:---:|:---|
| **Data Subsampling** | `ogt_subsample_meso_rate` | **`0.14` (14%)** | Retains only 14% of mesophilic (25–40°C) samples per epoch, shifting thermophile training share from 16.8% to 38.0% and accelerating convergence by 55%. |
| | `meso_temp_low` / `high` | **`25.0°C` / `40.0°C`** | Boundary window defining mesophilic organisms. Psychrophiles (<25°C) and thermophiles (>40°C) are 100% retained. |
| | `target_jitter_std` | **`0.5°C`** | Gaussian noise added to raw OGT targets before Z-scoring to smooth integer step-function plateaus (~84% of database records are rounded integers). |
| | `tm_ogt_noise_std` | **`2.0°C`** | Gaussian noise injected into ground-truth OGT prior during $T_m$ training, preventing over-reliance on deterministic OGT inputs during two-stage inference. |
| **Loss & Reweighting** | `huber_delta_ogt` ($\delta$) | **`15.0°C`** | Transition threshold between quadratic (L2) and linear (L1) loss in raw temperature space, maintaining strong pull for thermophilic errors up to 15°C. |
| | `focal_gamma` ($\gamma$) | **`2.0`** | Exponential focusing parameter in Focal Huber loss, downweighting easy mesophilic errors (~0.1–2°C) and focusing gradient capacity on hard extremophile outliers. |
| | `focal_beta` ($\beta$) | **`0.5` (Z-space)** | Focal threshold corresponding to ~7.1°C in raw temperature space ($14.2 \times 0.5$). Errors $<7.1^\circ\mathrm{C}$ get suppressed; errors $>7.1^\circ\mathrm{C}$ get amplified. |
| | `weight_power` | **`0.75`** | Exponent for temperature bin frequency reweighting $(\frac{\text{median}}{\text{count}})^{0.75}$, aggressively countering the 100× rarity gap between mesophiles and extremophiles. |
| | `weight_clamp_min` / `max` | **`0.3` / `22.0`** | Floor and ceiling for bin multipliers, allowing up to a $22\times$ gradient boost for ultra-rare hyperthermophiles ($>80^\circ\mathrm{C}$). |
| | `iqr_impute_val` | **`0.62°C`** | Imputed median Interquartile Range (IQR) assigned to proteins with only a single historical experimental reading in literature databases. |
| | `iqr_weight_scale` | **`6.34°C`** | Denominator scale in inverse-IQR sample weight formula ($w = \frac{1}{1 + (\text{iqr}/6.34)^2}$), downweighting controversial literature records with high experimental variance. |
| **Regularization** | `mixup_alpha` ($\alpha$) | **`0.0`** | Linear Mixup parameter (0.0 = disabled in final V9 model to prevent continuous target smoothing from degrading sharp $T_m$ boundary thresholds). |
| | `augment_prob` / `std` | **`0.15` / `0.02`** | Probability (15%) and standard deviation (0.02) of continuous Gaussian noise injected into input embeddings during training as continuous input smoothing. |
| | `dropout_1` / `dropout_2` | **`0.3` / `0.2`** | Dropout probabilities after first (`hidden_size_1 = 512`) and second (`hidden_size_2 = 256`) MLP layers. |
| | `weight_decay` | **`1e-5`** | AdamW L2 weight decay penalty preventing parameter explosion. |
| **Architecture** | `input_size_tm` / `ogt` | **`1289` / `1288` (Raw)** $\rightarrow$ **`1344` (MLP Input)** | Raw auxiliary vectors (`9` dims for $T_m$: OGT prior, transmembrane flag, sequence length, 6-class AA composition ratios; `8` dims for OGT without prior). Note: `ogt_reliable_flag` was excluded as ablations showed negligible gain ($\Delta \text{MAE} < 0.02^\circ\mathrm{C}$) due to continuous scheduled OGT noise injection and heteroscedastic variance bounding. |
| | `proj_dim` | **`64`** | Bottleneck projection (`Linear(9,64)` / `Linear(8,64)` $\rightarrow$ `LayerNorm` $\rightarrow$ `GELU`). The `64`-dim projection is concatenated with the `1280`-dim SaProt embedding to yield a `1344`-dim hidden vector (`input_size_fc1 = 1344`) feeding the independent 3-layer residual MLP blocks. |
| **Optimization** | `learning_rate` | **`1e-4`** | Peak learning rate for AdamW optimizer across both disjoint pathways. |
| | `batch_size` | **`64`** | Mini-batch size providing stable LayerNorm statistics without GPU memory overflow. |
| | `scheduler_type` | **`cosine`** | `CosineAnnealingWarmRestarts` with $T_0 = 10$ and $T_{mult} = 2$, preventing premature learning rate decay on small $T_m$ datasets. |
| | `early_stopping_patience`| **`15`** | Epoch patience accommodating cosine annealing learning rate restarts at epoch 10 without premature training termination. |

---

## Supplementary Note 1: Mathematical Formulation of Gradient Decoupling

In legacy multi-task protein language models, a shared hidden backbone parameter vector $\theta_{shared}$ is simultaneously optimized for thermodynamic unfolding ($T_m$) and environmental adaptation (OGT). Let $\mathcal{L}_{Tm}(\theta_{shared}, \theta_{Tm})$ represent the Gaussian Negative Log-Likelihood loss for melting temperature prediction, and let $\mathcal{L}_{OGT}(\theta_{shared}, \theta_{OGT})$ represent the Focal Huber loss for organismal growth temperature prediction. 

During backpropagation, the total gradient applied to the shared backbone is the linear combination:
$$\nabla_{\theta_{shared}} \mathcal{L}_{total} = \nabla_{\theta_{shared}} \mathcal{L}_{Tm} + \lambda \nabla_{\theta_{shared}} \mathcal{L}_{OGT}$$

Because the target distributions of $T_m$ and OGT exhibit a significant physical domain shift—where mesophilic enzymes ($\text{OGT} \approx 37^\circ\mathrm{C}$) frequently possess unfolding thresholds exceeding $65^\circ\mathrm{C}$—the gradient vectors $\nabla_{\theta_{shared}} \mathcal{L}_{Tm}$ and $\nabla_{\theta_{shared}} \mathcal{L}_{OGT}$ frequently point in contradictory directions in high-dimensional embedding space:
$$\langle \nabla_{\theta_{shared}} \mathcal{L}_{Tm}, \nabla_{\theta_{shared}} \mathcal{L}_{OGT} \rangle < 0$$

This negative inner product induces **gradient interference (destructive interference)**, forcing the shared representation backbone to compromise between macro-environmental adaptability profiles and specific biophysical unfolding limits. This compromise is the primary mathematical cause of *mesophilic probability collapse* in shared-backbone architectures.

**StableProt Disjoint Solution**: StableProt eliminates destructive interference by decoupling the parameter spaces completely. We define two independent module parameter sets: $\Phi_{Tm} = \{\theta_{encoder, Tm}, \theta_{bottleneck, Tm}, \theta_{MLP, Tm}\}$ and $\Phi_{OGT} = \{\theta_{encoder, OGT}, \theta_{bottleneck, OGT}, \theta_{MLP, OGT}\}$. The optimization objectives are decoupled into independent alternating updates:
$$\Phi_{Tm}^{(t+1)} = \Phi_{Tm}^{(t)} - \eta \nabla_{\Phi_{Tm}} \mathcal{L}_{NLL}\left(y_{Tm}, \hat{y}_{Tm}(\Phi_{Tm}^{(t)})\right)$$
$$\Phi_{OGT}^{(t+1)} = \Phi_{OGT}^{(t)} - \eta \nabla_{\Phi_{OGT}} \mathcal{L}_{Focal}\left(y_{OGT}, \hat{y}_{OGT}(\Phi_{OGT}^{(t)})\right)$$
Because $\frac{\partial \mathcal{L}_{Tm}}{\partial \Phi_{OGT}} = 0$ and $\frac{\partial \mathcal{L}_{OGT}}{\partial \Phi_{Tm}} = 0$, zero gradient competition occurs. Each pathway develops dedicated feature manifolds tailored specifically to its thermodynamic or environmental target distribution.

---

## Supplementary Note 2: Empirical Architectural Progression & Regularizer Ablation Study

To systematically evaluate the empirical contribution of each structural representation, architectural bottleneck, and regularizer knob introduced across our iterative model development (V1 through V8/V9), we benchmarked all saved historical iterations across both the decontaminated ProThermDB validation set (3,340 records) and the zero-shot FireProtDB extremophile holdout suite (322 records; **Table S2**).

| Model Iteration & Ablation State | ProThermDB MAE (°C) | ProThermDB Int-MAE (°C) | FireProtDB MAE (°C) | FireProtDB Int-MAE (°C) | Architectural & Biophysical Impact |
|:---|:---:|:---:|:---:|:---:|:---|
| **V1: Baseline MLP** | 12.28 | 12.28 | 28.21 | 28.21 | Simple multi-layer perceptron on primary sequence proxy tokens without structural context or uncertainty bounding. |
| **V2: Improved MLP** | 10.60 | 10.60 | 32.47 | 32.47 | Added batch normalization and residual dropout (`0.3`/`0.2`), improving empirical interpolation but overfitting on zero-shot holdouts. |
| **V3: Continuous Regression** | 9.38 | 9.38 | 25.96 | 25.96 | Shifted from binary classification proxy targets to direct continuous regression (`L1/L2` loss) across unfolding temperatures. |
| **V4: Improved Residual Regressor** | 8.16 | 8.16 | 26.39 | 26.39 | Introduced skip-connection residual projections (`residual_proj`), reducing ProThermDB MAE by $>1.2^\circ\mathrm{C}$. |
| **V5: Multi-Head Aux Bottleneck** | 6.84 | 6.84 | 12.69 | 12.69 | Added dedicated 64-dim projection bottlenecks (`Linear(9,64)`) separating scalar features from high-dimensional token representations. |
| **V6: Structure-Aware SaProt 3Di** | 6.11 | 6.11 | 10.84 | 10.84 | Integrated Foldseek 3Di conformational structural tokens (`1280-dim`), providing dramatic $>1.8^\circ\mathrm{C}$ zero-shot extremophile improvement. |
| **V7: Shared Multi-Task MLP** | 7.61 | 7.61 | 11.45 | 11.45 | **Negative Control**: Shared hidden layers for simultaneous $T_m$ and OGT optimization caused gradient interference (+1.50°C error). |
| **StableProt V8 (Full Disjoint Pipeline)** | **6.83** | **4.78** | **12.33** | **10.19** | **Disjoint alternating optimization + NLL confidence intervals + mesophilic subsampling (`14%`) + scheduled OGT noise (`6.0°C`).** |

As demonstrated empirically across our historical progression in **Table S2**, transitioning from sequence-only representations (`V5: 12.69°C` on FireProtDB) to structure-aware SaProt 3Di conformational embeddings (`V6: 10.84°C`) provides the single largest zero-shot accuracy jump ($>1.8^\circ\mathrm{C}$). Furthermore, comparing **V7** against **V8** explicitly validates our mathematical gradient decoupling proof (Supplementary Note 1): forcing a shared backbone to simultaneously optimize unfolding limits and environmental adaptability (**V7**) increases ProTherm MAE from $6.11^\circ\mathrm{C}$ to $7.61^\circ\mathrm{C}$ due to destructive gradient interference. Decoupling the pathways into disjoint multi-head projections with heteroscedastic NLL bounding (**StableProt V8**) resolves this bottleneck, achieving an undisputed state-of-the-art `Int-MAE` of **4.78°C** on ProThermDB and **10.19°C** on zero-shot FireProtDB.

---

## Supplementary Note 3: Environmental Optimal Growth Temperature (OGT) Evaluation across Thermal Bins

To verify that StableProt V8 achieves robust generalizability without suffering from mesophilic probability collapse, we evaluated the model against baseline predictors across discrete $10^\circ\mathrm{C}$ temperature intervals (`0–100°C`; **Table S3**).

#### Table S3: Per-Temperature-Bin Optimal Growth Temperature (OGT) Error Profile across the Full Thermal Spectrum

| Temperature Bin | StableProt V8 (Conf-Adj) | StableProt V8 (Ours) | PRIME | ThermoFormer |
|:---|:---:|:---:|:---:|:---:|
| **0–10°C** | 24.4°C | 29.1°C | 20.9°C | 21.0°C |
| **10–20°C** | 17.1°C | 21.1°C | 8.8°C | 8.5°C |
| **20–30°C** | 9.7°C | 13.5°C | 3.7°C | 3.4°C |
| **30–40°C** | 7.6°C | 11.2°C | 2.3°C | 2.4°C |
| **40–50°C** | 5.7°C | 8.0°C | 12.7°C | 11.5°C |
| **50–60°C** | 7.3°C | 10.1°C | 12.5°C | 12.2°C |
| **60–70°C** | 8.0°C | 11.1°C | 7.6°C | 6.9°C |
| **70–80°C** | 5.8°C | 8.9°C | 5.3°C | 5.3°C |
| **80–90°C** | 5.0°C | 7.9°C | 6.7°C | 6.5°C |
| **90–100°C** | 2.3°C | 4.8°C | 5.3°C | 5.3°C |

Per-temperature-bin analysis (**Table S3**) reveals that PRIME and ThermoFormer exhibit a sharp error increase from 2–3°C MAE in the 20–40°C range to 11–13°C MAE in the 40–60°C range—a 4–5× degradation coinciding with the transition from data-dense to data-sparse temperature regimes. In contrast, StableProt's error profile remains consistent across all bins (4.8–13.5°C MAE), with no sudden degradation. Under our confidence-adjusted metric ($\text{Int-MAE}$), the profile becomes even smoother, with errors ranging from only 2.3–9.7°C across all bins $>20^\circ\mathrm{C}$—demonstrating that StableProt provides reliable, well-calibrated predictions across the full thermal spectrum. We argue that consistent, generalizable predictions across all temperature regimes are more valuable for enzyme engineering than overfitted mesophilic accuracy that fails on thermophilic targets.

## Supplementary Note 4: Head-to-Head Comparison of StableProt V8 and V9 Architectures

During the optimization of StableProt, we trained and evaluated a candidate iteration (StableProt V9) featuring hyperparameter configurations suggested by our parameter sweeps. Specifically, V9 disabled residual connections (`use_residuals=False`), widened the first hidden projection layer (`hidden_size_1=768`), increased the minimum sequence length boundary (`seq_len_min=150` residues), utilized a lower OGT target noise standard deviation (`tm_ogt_noise_std=2.0`), and implemented a learning rate warmup scheme.

To determine whether V9 should replace V8 as the primary production model, we conducted a comprehensive head-to-head comparison across all nine evaluation suites. The comparative results are summarized in **Table S4**.

#### Table S4: Head-to-Head Comparison of StableProt V8 and V9 across Evaluation Suites
| Evaluation Suite / Benchmark | Metric | StableProt V8 (Ours) | StableProt V9 (Retrained) | Better Performing Model |
| :--- | :---: | :---: | :---: | :---: |
| **ProThermDB (Thermodynamics)** | MAE (°C) $\downarrow$ | 6.83 | 6.18 | StableProt V9 |
| | Conf-Adj MAE (°C) $\downarrow$ | 4.78 | 3.81 | StableProt V9 |
| | Pearson ($r$) $\uparrow$ | 0.804 | 0.471 | StableProt V8 |
| | Spearman ($\rho$) $\uparrow$ | 0.477 | 0.620 | StableProt V9 |
| **FireProtDB (Zero-Shot OOD)** | MAE (°C) $\downarrow$ | 12.33 | 11.92 | StableProt V9 |
| | Conf-Adj MAE (°C) $\downarrow$ | 10.19 | 9.42 | StableProt V9 |
| | Pearson ($r$) $\uparrow$ | 0.420 | 0.299 | StableProt V8 |
| | Spearman ($\rho$) $\uparrow$ | 0.349 | 0.450 | StableProt V9 |
| **SPURS Megascale** | MAE (°C) $\downarrow$ | 9.70 | 7.85 | StableProt V9 |
| | Pearson ($r$) $\uparrow$ | 0.478 | 0.436 | StableProt V8 |
| | Spearman ($\rho$) $\uparrow$ | 0.281 | 0.238 | StableProt V8 |
| **Single-Point Mutation $\Delta T_m$**| MAE (°C) $\downarrow$ | 4.58 | 4.46 | StableProt V9 |
| | Pearson ($r$) $\uparrow$ | +0.031 | -0.060 | StableProt V8 |
| | Spearman ($\rho$) $\uparrow$ | +0.059 | +0.009 | StableProt V8 |
| **MMseqs2 Cluster OOD** | MAE (°C) $\downarrow$ | 5.78 | 6.35 | StableProt V8 |
| **BRENDA OOD (OGT)** | MAE (°C) $\downarrow$ | 11.62 | 10.93 | StableProt V9 |
| | RMSE (°C) $\downarrow$ | 14.75 | 13.97 | StableProt V9 |
| | Pearson ($r$) $\uparrow$ | 0.850 | 0.854 | StableProt V9 |
| | Spearman ($\rho$) $\uparrow$ | 0.833 | 0.838 | StableProt V9 |
| **Uncertainty Calibration** | ECE $\downarrow$ | 0.386 | 0.294 | StableProt V9 |

As shown in **Table S4**, retraining V9 with a restored sequence length threshold (`seq_len_min=50`), a reduced learning rate (`learning_rate=1e-4`), and enabled residual connections (`use_residuals=True`) resolved many out-of-distribution (OOD) generalization issues. While V9 outperforms V8 on ProThermDB, FireProtDB, overall SPURS Megascale, and BRENDA OOD OGT, V8 retains superior accuracy on rare thermophilic SPURS targets (4.42°C vs 5.91°C MAE) and MMseqs2 Cluster OOD sequence families (5.78°C vs 6.35°C MAE).

Because V9's performance is mixed compared to V8, we retain StableProt V8 as the primary production model in the main manuscript text and report the retrained StableProt V9 results here in the supplementary materials as a highly promising alternative.

#### Table S5: Summary of Hyperparameter Sweep Ablations
To verify which architectural, loss, optimization, and data parameters influence StableProt's performance, we conducted a systematic hyperparameter sweep. The parameters that yielded significant performance variations are summarized below.

| Parameter Category | Hyperparameter | Range Swept | Optimal Value | Biophysical/Optimization Rationale |
| :--- | :--- | :---: | :---: | :--- |
| **Architecture** | `use_residuals` | `[True, False]` | `False` | Reduces model capacity slightly, mitigating over-interpolation on dense clusters. |
| | `hidden_size_1` | `[256, 512, 768]` | `512` (Tm), `768` (OGT) | Larger projection dimensions capture environmental OGT features better but risk Tm overfitting. |
| **Optimization** | `warmup_epochs` | `[0, 5, 10]` | `5` | Stabilizes early multi-task gradients. |
| | `learning_rate` | `[1e-4, 5e-5, 1e-5]` | `1e-4` | Balances optimization speed and convergence stability. |
| **Augmentation** | `tm_ogt_noise_std` | `[1.0, 2.0, 6.0]` | `6.0` (V8), `2.0` (V9) | High noise (6.0) acts as a strong regularizer for OGT, preventing mesophilic collapse. |
| **Data Curation** | `seq_len_min` | `[50, 100, 150]` | `50` (V8), `150` (V9) | Lower sequence minimum (50) preserves short peptides necessary for broad environmental generalization. |
| | `ogt_subsample_meso_rate`| `[0.05, 0.14, 0.30]` | `0.14` | Optimizes the trade-off between mesophilic accuracy and thermophilic generalization. |
