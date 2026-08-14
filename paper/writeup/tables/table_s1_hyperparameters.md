# Supplementary Table S1: Curated Architectural and Training Hyperparameters

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
