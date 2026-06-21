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
