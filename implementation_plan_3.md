# StableProt V7 — Final Execution Plan

> After 3 rounds of debate (Claude vs DeepSeek), this is the agreed plan. No more changes.

---

## Architecture

```
        SaProt 1.3B (FROZEN)
   ┌─────────────────────────────┐
   │  Tm seqs:  full SA tokens   │  ← structures from ESMFold/PDB
   │  OGT seqs: seq-only (#mask) │  ← no structure needed
   └─────────────┬───────────────┘
                 │ 2560-dim
         ┌───────┴───────┐
         │ Shared Backbone│
         │ Linear 2560→H1 │
         │ BN + ReLU + Drop│
         │ Linear H1→H2    │
         │ BN + ReLU + Res │
         └───────┬───────┘
                 │ H2-dim
         ┌───────┴───────┐
    ┌────┴────┐    ┌────┴────┐
    │ OGT Head│    │ Tm Head │
    │  → 1    │    │  → 1    │
    └─────────┘    └─────────┘

Loss_total = 0.03 × Huber(OGT) + Huber_weighted(Tm)
```

**Key decisions (locked in)**:
- SaProt 1.3B frozen — no fine-tuning
- OGT loss scaled ×0.03 to prevent gradient domination (33:1 data ratio)
- Tm weighted Huber: `weight = min(sqrt(median_count / bin_count), 8.0)`, 5°C bins
- 5-seed ensemble with early stopping

> [!IMPORTANT]
> **SaProt variant choice needed**: `SaProt_1.3B_AF2` (204 downloads) vs `SaProt_1.3B_AFDB_OMG_NCBI` (5,107 downloads). Recommend AFDB+OMG+NCBI — trained on larger structural database, better for rare protein families.

---

## Phase 0: BacDive OGT Label Replacement
**Time: 4 hours | Depends on: nothing**

1. Download BacDive organism→OGT mapping (API or bulk CSV)
2. Build lookup: `organism_name → BacDive_OGT`
3. For each OGT training sequence:
   - If organism in BacDive AND `|original_OGT - BacDive_OGT| < 20°C`: replace label
   - If organism in BacDive AND `|difference| ≥ 20°C`: flag for manual review (likely taxonomy mismatch)
   - If organism NOT in BacDive: keep original label
4. Log replacement statistics (how many replaced, coverage %)

**Output**: Cleaned OGT label file (`ogt_labels_bacdive_corrected.csv`)

**Why 20°C gate**: BacDive maps OGT to organisms, not proteins. A >20°C disagreement likely means organism name mismatch, not a bad label. Blind overwrite without this gate risks introducing errors.

---

## Phase 1: Data Re-split + CD-HIT Decontamination
**Time: 2 hours | Depends on: nothing (parallel with Phase 0)**

1. Load current 28,739 Tm training sequences
2. Temperature-stratified split: ~25,900 train + ~2,800 val
   - Use 5°C bins for stratification to preserve extremophile representation
3. Keep existing 2,007 test set unchanged
4. Run CD-HIT at 30% identity across ALL splits:
   - Train vs Val: remove any val sequence with >30% identity to train
   - Train vs Test: remove any test sequence with >30% identity to train
   - Val vs Test: remove any overlaps
5. Report final split sizes and per-bin sample counts

**Output**: `prepared_data_v7_splits.pt` with clean train/val/test

---

## Phase 2: SaProt 1.3B Embedding Generation
**Time: 1-2 days HPC | Depends on: Phase 0 (OGT labels), Phase 1 (splits)**

Generate frozen SaProt 1.3B embeddings for ALL sequences:

| Dataset | Count | Mode | Structure Source |
|---------|-------|------|-----------------|
| Tm train | ~25,900 | Structure-aware (SA tokens) | ESMFold PDB (HPC — in progress) |
| Tm val | ~2,800 | Structure-aware | ESMFold PDB |
| Tm test | 2,007 | Structure-aware | ESMFold PDB |
| OGT train | 941,041 | **Sequence-only** (`#` mask) | None needed |
| ProThermDB eval | 5,460 | Structure-aware where available, seq-only fallback | ESMFold PDB |
| FireProtDB eval | 324 | Structure-aware where available, seq-only fallback | ESMFold PDB |

**Total**: ~975K sequences × 2560-dim = ~9.3 GB embeddings

**Compute estimate**:
- SaProt 1.3B inference: ~200-300K sequences/day per GPU (batched)
- With 4 HPC GPUs: ~1-2 days total

**Output**: `prepared_data_v7_saprot1.3b.pt`

---

## Phase 3: Train V7 Shared Backbone
**Time: 4-6 hours | Depends on: Phase 2**

1. Implement `MultiHeadSaProtV7` with shared backbone + separate heads
2. Loss: `L = 0.03 × Huber(OGT_pred, OGT_true) + Σ w_b × Huber(Tm_pred_b, Tm_true_b)`
3. Weighting: `w_b = min(sqrt(median_count / bin_count), 8.0)`, 5°C bins from 25-100°C
4. Optimizer: Adam (lr from Phase 4 search, or default 1e-4)
5. Scheduler: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)
6. Early stopping: patience 12 on 2,800-sample val set
7. 5-seed ensemble

**Output**: 5 model checkpoints in `experiments/src/training/v7_shared/results/`

---

## Phase 4: Optuna Hyperparameter Search
**Time: 2-3 GPU days | Depends on: Phase 2**

Search space:
```python
{
    'lr': (1e-5, 1e-3, log),
    'weight_decay': (1e-6, 1e-3, log),
    'dropout': (0.1, 0.5),
    'hidden1': [256, 512, 768],
    'hidden2': [128, 256, 384],
    'batch_size': [32, 64, 128],
    'ogt_loss_scale': (0.01, 0.1),
    'weight_clamp': (4.0, 12.0),
}
```

Protocol:
- 1 seed per trial, 50-100 trials
- Objective: minimize val MAE on 2,800-sample val set
- Top 3 configs → 5-seed ensemble each → select best

**Output**: Best hyperparameter config JSON

---

## Phase 5: Evaluation
**Time: 1 hour | Depends on: Phase 3 or 4**

Evaluate on:
1. **ProThermDB** (5,460 sequences) — primary in-distribution benchmark
2. **FireProtDB** (324 sequences) — out-of-distribution benchmark
3. **FLIP Meltome** (decontaminated) — external standardized benchmark
   - Download FLIP test split
   - CD-HIT at 40% against all training data
   - Report results only on non-overlapping subset
   - Transparently state original vs decontaminated sample count

Metrics: MAE, RMSE, PCC, Spearman, R², per-bin MAE (5°C bins)

---

## Phase 6: Isotonic Calibration
**Time: 1 hour | Depends on: Phase 5**

1. 5-fold cross-calibration on 2,007 test set
2. Train isotonic regression: predicted_Tm → calibrated_Tm
3. Report BOTH raw and calibrated metrics
4. **Explicitly state**: calibration fixes MAE/bias but NOT PCC/Spearman

---

## Rejected Proposals (Final)

| Proposal | Reason | Who Proposed |
|----------|--------|-------------|
| BRENDA Topt as training data | Topt ≠ Tm (5-15°C offset), introduces systematic bias | DeepSeek R2 |
| BRENDA as validation | Adds confusion, not needed | DeepSeek R2 |
| Focal loss | Untested for regression, instability risk | Claude R1 |
| SaProt 3B | Doesn't exist | Claude R1 |
| Fine-tune SaProt layers | Overfitting risk with 28K samples, unnecessary | DeepSeek R2 |
| GradNorm | Over-engineering; fixed 0.03 scale sufficient | Claude R3 |
| Gaussian embedding augmentation | Doesn't add protein family diversity | Claude R1, rejected R2 |
| Stochastic depth | Architecture too shallow (2 layers) | DeepSeek R2 |
| OGTFinder | Nice-to-have but not essential for V7 | DeepSeek R3 |
| 88-sample val set | Too small for early stopping | All agreed |
| EsmTemp benchmark | No pre-trained checkpoint available | Claude R1 |

---

## Success Criteria

| Benchmark | V6 (Current) | V7 Target | SOTA Threshold |
|-----------|:------------:|:---------:|:--------------:|
| ProThermDB MAE | 5.74°C | **< 4.5°C** | Beat TemBERTure 5.49°C |
| FireProtDB MAE | 11.78°C | **< 10.0°C** | Already SOTA |
| ProThermDB PCC | 0.85 | **> 0.88** | Beat TemBERTure 0.86 |
| FLIP Spearman | — | **> 0.65** | Competitive |

If ProThermDB MAE < 5.0°C AND FireProtDB MAE < 11.0°C → **undisputed SOTA**.
