# StableProt V7 — Complete Pipeline Audit

End-to-end trace: Data Cleaning → Splitting → Embeddings → Training → Evaluation.
All claims verified against actual code. Attribution: [MIMO], [CLAUDE], or both.

---

## 🔴 CRITICAL BUGS

### C1. [CLAUDE] ProThermDB Eval Contaminated — 38.8% Training Data in Benchmark
**File:** [evaluate_all_models_protherm.py:408-438](file:///home/bibhu/Documents/temstampto/experiments/src/eval/evaluate_all_models_protherm.py#L408-L438)

| Subset | Count | % of Eval |
|---|---|---|
| In V7 train_tm | **2,120** | **38.8%** |
| In V7 val_tm | 94 | 1.7% |
| In V7 test_tm | 1,894 | 34.7% |
| NOT in any V7 split | 1,352 | 24.8% |

Reported manuscript numbers (MAE 5.74 V6, 6.51 V7) are inflated. V6 eval has same problem (line 363 maps across all splits). Internal `test_tm` (2,007 seqs) IS clean.

### C2. [MIMO] Gradient Accumulation ≠ Alternating Optimization
**File:** [train.py:193-226](file:///home/bibhu/Documents/temstampto/experiments/src/training/v7_shared/train.py#L193-L226)

Single `zero_grad()` → `tm_loss.backward()` → `ogt_loss.backward()` → `optimizer.step()`. Gradients accumulate jointly. V5/V6 do true alternating (separate zero_grad/backward/step per task). Manuscript says "alternating task optimization."

### C3. [MIMO+CLAUDE] Three Manuscript Factual Errors
- **"layer normalization"** → Code: `nn.BatchNorm1d(hidden1)` (L68)
- **"AdamW optimizer"** → Code: `optim.Adam(...)` (L162)
- **"alternating task optimization"** → Code: joint gradient accumulation (see C2)

### C4. [MIMO+CLAUDE] evaluate_v7_joint.py Dead Import — Script Crashes
**File:** [evaluate_v7_joint.py:14](file:///home/bibhu/Documents/temstampto/experiments/src/eval/evaluate_v7_joint.py#L14)
```
from experiments.src.training.v7_transfer.model import StableProtV7
```
`v7_transfer/` does not exist. ImportError on execution.

### C5. [MIMO+CLAUDE] OGT Eval and Tm Eval Use DIFFERENT Trained Models
- Tm eval: `v7_shared/results/seed{s}/best_model.pt`
- OGT eval: `v7_shared/ogt_benchmark_results/seed{s}/best_model.pt`

MD5 checksums differ. OGT eval also loads Optuna config for hidden dims; Tm eval uses defaults. These are completely different model weights — possibly trained with different hyperparameters.

### C6. [MIMO+CLAUDE] CD-HIT Config=40%, Docstring/Print=30%
**File:** [phase1_resplit_cdhit.py:32](file:///home/bibhu/Documents/temstampto/experiments/src/data/phase1_resplit_cdhit.py#L32)
```python
CDHIT_IDENTITY = 0.40  # But docstring line 7 and print line 173 say "30%"
```
Anyone watching the output is told 30% identity when the actual threshold is 40%.

---

## 🔴 CODE BUGS (Verified)

### B1. [MIMO] Dead Code — batch_weights Overwritten
**File:** [train.py:197-209](file:///home/bibhu/Documents/temstampto/experiments/src/training/v7_shared/train.py#L197-L209)

First `batch_weights` computed with sequential indices (meaningless), immediately overwritten by correct label-based version.

### B2. [MIMO+CLAUDE] Model Default + Comment Wrong
**File:** [train.py:60-62](file:///home/bibhu/Documents/temstampto/experiments/src/training/v7_shared/train.py#L60-L62)
```python
"""SaProt 1.3B (frozen, 2560-dim)"""   # SaProt 1.3B outputs 1280-dim
def __init__(self, input_dim=2560, ...):  # CONFIG uses 1280
```

### B3. [MIMO] No Gradient Clipping in V7
V4, V5, V6 all use `clip_grad_norm_(model.parameters(), 1.0)`. V7 does not. grep confirms zero `clip_grad` references in V7. Multi-task training with different loss scales = gradient instability risk.

### B4. [MIMO] BacDive Errors Cached as None — Indistinguishable
**File:** [phase0_bacdive_ogt.py:117-119](file:///home/bibhu/Documents/temstampto/experiments/src/data/phase0_bacdive_ogt.py#L117-L119)

API errors → `cache[tax_id] = None`. Same as "no data available." If BacDive was temporarily down, those organisms never get re-queried. Should use a sentinel like `"_error"`.

### B5. [MIMO] Optuna Has Same Gradient Accumulation Bug
**File:** [phase4_optuna_search.py:118-144](file:///home/bibhu/Documents/temstampto/experiments/src/training/v7_shared/phase4_optuna_search.py#L118-L144)

Identical pattern: `zero_grad()` → `tm.backward()` → `ogt.backward()` → `step()`. Hyperparameters were optimized around the buggy training loop.

### B6. [MIMO] CD-HIT-2D Function Default Doesn't Match Usage
```python
def run_cdhit_2d(..., identity=0.30):  # Default 0.30
    ...
run_cdhit_2d(..., identity=CDHIT_IDENTITY)  # Actually called with 0.40
```

### B7. [MIMO] CD-HIT -G 0 Mentioned in Docstring But Not Used
Docstring line 72: "Use -G 0 for local alignment." Actual command omits `-G 0`. Uses default global alignment.

---

## 🟡 TRAINING DESIGN ISSUES (Verified)

### T1. [MIMO] Huber δ=1.0 Too Aggressive
V6: `delta=5.0`. V7: `delta=1.0`. Optuna hardcodes `delta=1.0` (not searched over). With δ=1.0, any error >1°C is treated linearly (L1-like). For typical 5-10°C errors, δ=5.0 more appropriate.

### T2. [MIMO] No OGT Loss Weighting — 75% Mesophilic Imbalance
Tm gets `sqrt(median/count)` per-sample weights. OGT gets nothing — just `huber(...).mean() * 0.03`. 74.9% of OGT data is 25-45°C. Shared backbone biased toward mesophilic representations.

### T3. [MIMO+CLAUDE] OGT Sees Only ~2.7% of Data Per Epoch
OGT has ~941K samples. Tm has ~26K. OGT iterator resets per epoch, produces only as many batches as Tm (~406 batches × 64 = ~26K samples). Per epoch: 26K/941K = 2.7%.

### T4. [MIMO] No OGT Validation
`# Validation (Tm only)` at L236. Early stopping and model selection based solely on Tm MAE. OGT head could be overfitting, underfitting, or collapsing — no way to know.

### T5. [MIMO+CLAUDE] ogt_loss_scale 10× Reduction Never Justified
V5/V6: `ogt_loss_weight = 0.3`. V7: `ogt_loss_scale = 0.03`. May compensate for gradient accumulation bug (C2), but never ablated or explained.

### T6. [MIMO+CLAUDE] No Learning Rate Warmup
No warmup scheduler in V4, V5, V6, or V7. `CosineAnnealingWarmRestarts` starts at full LR. BatchNorm running stats noisy in early epochs.

### T7. [MIMO] Optuna Huber Delta Not Searched
Hardcoded `delta=1.0` in search script. Not in Optuna search space. If δ=5.0 better, Optuna can't find it.

### T8. [MIMO] Optuna Fixed Seed=42 for All Trials
Every trial uses same seed. Different seeds could yield different optimal configs.

### T9. [MIMO] Optuna Searches 1 Seed × 100 Epochs — Final Uses 5 Seeds × 150 Epochs
Config found under reduced conditions may not transfer.

---

## 🟡 UNUSED FEATURES & MISSED OPPORTUNITIES (Verified)

### F1. [CLAUDE] OGT as Input Feature to Tm Head — Not Used
91.8% of Tm sequences have reliable OGT (`ogt_reliable` flag). Host OGT is a strong prior for Tm (Tm ≈ OGT + 20-30°C). Concatenate `[embedding, ogt, ogt_reliable]` → 1282-dim. Trivial cost, high potential gain.

### F2. [CLAUDE] Transmembrane Protein Flag — Not Used
`tmhmm_tm_binary`: 11% of training data are transmembrane proteins. Different stability profiles (lipid bilayer stabilization). Treated identically to soluble proteins.

### F3. [MIMO] ogt_reliable Flag Not Used
Flag exists in data but never incorporated into loss weighting or input features.

### F4. [CLAUDE] No AA Composition Features
Charged residue fraction, hydrophobic fraction, proline content, cysteine pairs, (Ile+Val)/(Ile+Val+Leu) — all trivially computable, historically proven for thermostability. Not used.

### F5. [CLAUDE] No Sequence Length Feature
Available for free, never used.

### F6. [CLAUDE+MIMO] Mean Pooling Only — No CLS/Attention/Max
Mean pooling discards spatial arrangement info (which positions are stabilizing). `AttentionPool` exists in V5 but not ported to V7. CLS token available but unused. Mean+Max+StdDev (3840-dim) would capture richer distributional info.

### F7. [MIMO] No Test-Time Augmentation
No reverse-sequence averaging. Cheap uncertainty estimate from forward/reverse disagreement.

### F8. [CLAUDE] Ensemble Uses Simple Mean — Not Weighted
5-seed ensemble averages predictions. Seeds have different val MAEs. Inverse-error weighting or stacking would improve.

### F9. [MIMO] No Organism-Level OGT Consistency
Each protein independently predicts OGT. OGT is organism-level — consistency regularization across same `tax_id` proteins could help.

### F10. [MIMO] No Cross-Validation
Single fixed split (seed=42). K-fold would give more reliable performance estimates.

---

## 🟠 REGRESSION ALTERNATIVES (Suggestions)

### R1. [CLAUDE] Quantile Regression — Confidence Intervals for Free
Predict 10th/50th/90th percentiles with pinball loss. Three heads, same backbone.

### R2. [CLAUDE] Gaussian NLL — Heteroscedastic Uncertainty
Predict mean + log-variance. Model learns per-protein uncertainty.

### R3. [CLAUDE] Ordinal Regression — Temperature as Ordered Classes
Discretize Tm into 2-5°C bins. Predict cumulative probabilities. Better for imbalanced tails.

### R4. [CLAUDE] OGT Organism-Level Consistency Regularization
Penalize variance in OGT predictions across proteins from same `tax_id`.

### R5. [MIMO] No Label Smoothing / Noise Injection
Tm measurements have ~1-3°C experimental uncertainty. Adding Gaussian noise to targets during training could improve generalization.

---

## 🟡 DATA PIPELINE ISSUES (Verified)

### D1. [CLAUDE] SaProt Truncation — ~1022 AA Cutoff, Test Set Disproportionately Hit
SaProt tokenizer treats `M#` as 1 token (verified experimentally). 1024 max_length = ~1022 AAs.
String truncation at L86 (`s[:max_length * 2]`) is **redundant** — tokenizer's `truncation=True` is the binding constraint.

| Split | >1022 AA | % Truncated |
|---|---|---|
| train_tm | 2,132 | 8.2% |
| val_tm | 175 | 9.7% |
| **test_tm** | **359** | **17.9%** |

### D2. [MIMO] ESMFold Truncates at 1000 AA
`seq_trunc = seq[:1000]` in generate_tm_structures.py:90. Structure-aware embeddings lose C-terminal info for proteins >1000 AA.

### D3. [MIMO] BacDive-Corrected OGT Not Auto-Piped to Phase 1
Phase 0 outputs `ogt_labels_bacdive_corrected.pt`. Phase 1 reads `prepared_data_v4_saprot.pt`. Manual intervention needed (but `ogt_bacdive_corrected: True` in V7 data confirms it WAS done).

### D4. [MIMO] BacDive Sanity Gate 20°C May Reject Legitimate Corrections
If BacDive says 80°C but original says 55°C (Δ=25°C > gate), correction rejected. Could miss legitimate thermophile corrections.

### D5. [MIMO] No Embedding Quality Validation
No NaN checks, no all-zero detection, no norm distribution analysis after generation.

### D6. [MIMO] merge_all() Doesn't Validate Embedding-Label Alignment
Replaces embeddings by path order. No explicit alignment check.

### D7. [MIMO] MMseqs2 at 30% vs CD-HIT at 40% — Inconsistent Homology Thresholds
Phase 5 (OGT): MMseqs2 `--min-seq-id 0.3`. Phase 1 (Tm): CD-HIT `identity=0.40`.

### D8. [MIMO] External OGT Test Set Capped at 5,000 Sequences
`test_set = survivors[:5000]`. With 941K training sequences, could use more.

### D9. [MIMO] External OGT Uses Fragile Organism Name Matching
Spelling variants, synonyms, subspecies annotations could cause false positives/negatives.

---

## 🟢 MINOR / INFRA (Verified)

### M1. [MIMO] No Data Versioning — No checksums on data files.
### M2. [MIMO] No Reproducibility Logging — No git hash, PyTorch/CUDA version logged.
### M3. [MIMO] No Mixed Precision in Training — phase2 uses autocast, training doesn't.
### M4. [MIMO] Embedding Dimension Hardcoded — Should read from `model.config.hidden_size`.
### M5. [MIMO] `torch.load` without `weights_only` — Security warning in some scripts.
### M6. [MIMO] V7 Uses Sequence-Only, Not Structure-Aware — Pipeline exists but unused.
### M7. [MIMO] Foldseek Token Length Mismatch Falls Back to Mask — Handled but lossy.

---

## ❌ CLAIMS PROVEN WRONG

| Claim | Why Wrong |
|---|---|
| **P2-2** "1024 tokens = 512 AAs" | SaProt tokenizer treats `M#` as 1 token. 1024 tokens ≈ 1022 AAs. |
| **PE-4** "Eval loads with default dropout" | `model.eval()` disables dropout. Constructor values don't affect inference. |
| **P1-6** "Val split min 1 is a bug" | Intentional stratification — ensures every bin has ≥1 val sample. |
| **CC-4** "OGT-Tm overlap is a concern" | 37 sequences overlap out of 967K combined = 0.004%. Negligible. |
| **P2-4 wording** "Same composition → same embedding" | Transformer gives context-dependent per-residue embeddings. Mean pooling loses spatial info, not sequence identity. |

---

## Summary: Priority Ranking

| Priority | Item | Effort | Impact |
|---|---|---|---|
| 🔴 CRITICAL | C1 ProThermDB eval contamination | Rerun eval | **Manuscript numbers wrong** |
| 🔴 CRITICAL | C3 Manuscript LN/AdamW/alternating errors | Trivial | Factual errors in paper |
| 🔴 CRITICAL | C5 OGT vs Tm eval use different models | Medium | Inconsistent benchmarks |
| 🔴 HIGH | C2 Gradient accumulation bug | 1 line | Changes training dynamics |
| 🔴 HIGH | B3 No gradient clipping (V4-V6 had it) | 1 line | Training stability |
| 🔴 HIGH | T1 Huber δ=1.0 (V6 used 5.0) | Config | Optimization target |
| 🔴 HIGH | C6 CD-HIT 40% vs printed 30% | 1 line | Misleading output |
| 🟡 HIGH | F1 OGT as Tm input feature | Low | Strong prior, high gain |
| 🟡 HIGH | T2 OGT loss weighting | Low | Shared backbone bias |
| 🟡 HIGH | T3 OGT sees only 2.7% per epoch | Medium | OGT head weak |
| 🟡 HIGH | R1 Quantile regression | Medium | Uncertainty for free |
| 🟡 MED | F6 Better pooling (CLS/attention) | Medium | Medium-High |
| 🟡 MED | T5 ogt_loss_scale 10× drop | Ablation | Medium |
| 🟡 MED | F4 AA composition features | Low | Medium |
| 🟡 MED | B5 Optuna same gradient bug | 1 line | Config validity |
| 🟢 LOW | F2 TM protein flag | Low | Low-Medium |
| 🟢 LOW | F5 Sequence length feature | Trivial | Low |
| 🟢 LOW | F8 Weighted ensemble | Low | Low |
| 🟢 LOW | R5 Label smoothing | Low | Low |
| 🟢 LOW | T4 OGT validation | Low | Low |

---

## 🔧 Alternative Design Decisions That Could Improve Accuracy

Every decision below is a "knob" currently set to one value. Changing it could improve (or worsen) results. Organized by pipeline stage.

### DATA AGGREGATION — How Tm Consensus Is Computed

**Current: Median aggregation** per protein from Meltome/TemBERTure measurements.

Meltome stats (25,370 multi-measurement proteins): median intra-protein std = 0.00°C, mean = 0.50°C, max = 12.41°C.

| Alternative | How | Why It Could Help |
|---|---|---|
| **Weighted median by measurement confidence** | Weight each measurement by inverse experimental error or replicate count | High-quality experiments (many replicates) should contribute more. Currently all measurements equal. |
| **Use measurement range as a feature** | Pass `[tm_median, tm_std, tm_count]` to model | Model learns that proteins with high measurement variance are noisier. 433 proteins have std>5°C — unreliable labels. |
| **Source-aware aggregation** | Weight Meltome higher than TemBERTure (or vice versa) | Different methodologies (TPP mass-spec vs literature-curated) may have systematic biases. 4,847 proteins from TemBERTure only. |
| **Exclude high-variance proteins** | Remove proteins with intra-sequence std > 5°C (433 = 1.7%) | These are noisy labels that confuse the model. Small cost. |
| **Mean instead of median** | Mean is better MLE for Gaussian noise; median is more robust | Avg |mean-median| = 0.04°C, max = 4.58°C. Usually irrelevant, but for ~35 extreme outlier proteins, could matter. |

### DATA CLEANING — Which Proteins Are Included/Excluded

| Alternative | How | Why It Could Help |
|---|---|---|
| **Remove Tm < OGT proteins** | 1,025 proteins (4.0%) have Tm below host OGT — suspicious (avg deficit 15.1°C, max 32.1°C) | Likely measurement errors, IDPs, or chaperone-dependent. Teaches wrong Tm-OGT relationships. |
| **Remove very short proteins (<50 AA)** | 21 proteins <50 AA, 638 <100 AA | Peptide biophysics differs. SaProt embeddings may be unreliable for short sequences. |
| **Use chaperone_client flag** | Flag exists in data but unused | Chaperone clients may have artificially low in-vitro Tm. Downweight or exclude. |
| **Filter by OGT reliability** | `ogt_reliable` flag exists, unused | Unreliable OGT (inferred from taxonomy) adds noise to the Tm-OGT relationship. |
| **Separate transmembrane proteins** | `tmhmm_tm_binary`: 11% of data | TM proteins unfold differently. Single model adds label noise. Separate model or conditioning flag. |

### LOSS FUNCTION — What Error Gets Optimized

| Alternative | Current | Why Change |
|---|---|---|
| **Huber delta** | δ=1.0 (V7), was δ=5.0 (V6) | δ=1.0 treats any error >1°C as linear. Tm std=11.5°C — most errors >1°C. δ=5.0 or δ=10.0 keeps quadratic penalty for moderate errors. |
| **Loss function** | Huber | MSE, Quantile (uncertainty), LogCosh (smooth+robust), Tukey biweight (extreme robustness). |
| **Weight formula** | `sqrt(median/count)` | Bin 25-30°C: 4 samples, weight=11.1×. Alternatives: `cbrt`, `log1p`, or capped `min(sqrt(...), 3.0)`. |
| **Weight capping** | None | One bad label in the 4-sample bin is catastrophic at 11× weight. Cap at 3-5×. |
| **Source-aware weighting** | All sources equal | Weight curated ProThermDB higher than TPP-derived Meltome or NLP-extracted TemBERTure. |

### SAMPLE WEIGHTING & TRAINING STRATEGY

| Alternative | How | Why It Could Help |
|---|---|---|
| **OGT-aware Tm weighting** | Upweight proteins with unusual Tm-OGT gap | Extreme Tm-OGT cases are informative edge cases. Currently treated equally. |
| **Curriculum learning** | Train on "easy" proteins first, add hard ones later | Prevents early instability from noisy extreme-temp labels. |
| **Hard example mining** | After N epochs, upweight proteins with highest prediction error | Forces model to focus on errors rather than optimizing already-correct predictions. |

### ARCHITECTURE CHOICES

| Alternative | Current | Why Change |
|---|---|---|
| **Pooling** | Mean pool | CLS token, attention pooling, max pooling, or mean+max+std (3840-dim). |
| **Normalization** | BatchNorm | LayerNorm (batch-independent, more stable for small batches). |
| **Depth** | 2 shared layers (1280→512→256) | 3-4 layers may capture more complex nonlinear Tm relationships. |
| **Skip connections** | None | Residual connections in shared backbone. Standard practice. |
| **Head independence** | Shared BN, linear output | 1-2 private layers per head for task specialization. |
| **Auxiliary inputs** | 1280-dim embedding only | Concatenate: `[embedding, ogt, ogt_reliable, tmhmm, seq_len, charged_frac, hydrophobic_frac]`. |

### OPTIMIZER & SCHEDULE

| Alternative | Current | Why Change |
|---|---|---|
| **Optimizer** | Adam | AdamW (proper weight decay), LAMB, AdaFactor. |
| **LR warmup** | None — full LR from epoch 1 | Linear warmup 5-10 epochs. Standard with BatchNorm. |
| **Gradient clipping** | None (V7) | Restore `clip_grad_norm_(1.0)` from V4-V6. |
| **Multi-task strategy** | Joint accumulation (buggy) | True alternating, GradNorm, PCGrad, uncertainty weighting. |
| **OGT data coverage** | Sees 2.7% per epoch | Full OGT epoch or weighted random sampling across all 941K. |

### EVALUATION CHOICES

| Alternative | Current | Why Change |
|---|---|---|
| **Cross-validation** | Single split, seed=42 | 5-fold CV gives confidence intervals. |
| **Ensemble** | Simple mean of 5 seeds | Inverse-MAE-weighted, stacking, or uncertainty-weighted. |
| **Metrics** | Overall MAE/RMSE/PCC/R² | Temperature-stratified MAE per 10°C bin. Overall MAE hides extreme-temp failures. |
| **External benchmark** | ProThermDB (contaminated) | Use clean `test_tm` only. Add FireProtDB, BRENDA OGT benchmark. |
| **Consistency check** | Not done | Verify Tm_pred > OGT_pred. Flag biologically implausible predictions. |

### OGT DATA HANDLING

| Alternative | Current | Why Change |
|---|---|---|
| **OGT aggregation** | Median of BacDive + original | Use BacDive exclusively when available, or weighted average by source confidence. |
| **Integer OGT smoothing** | 83.9% of OGT values are exact integers | Add Gaussian noise (±1°C) during training to prevent learning integer artifacts. |
| **Consistency regularization** | Each protein predicts OGT independently | Penalize variance of OGT predictions within same `tax_id`. |
| **OGT loss scale** | 0.03 (10× lower than V6's 0.3) | Never ablated. Auto-balance with GradNorm instead. |
