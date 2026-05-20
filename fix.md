# StableProt V2 — Fix Tracker

> Last updated: 2026-05-20 (Session 2)

## Current V6 Status

| Metric | Old V6 (broken) | Fixed V6 | V5 (ProtT5) |
|--------|-----------------|----------|-------------|
| ProThermDB Tm MAE | 5.40°C | **5.40°C** ✅ | 7.29°C |
| FireProt OOD MAE | collapsed (Std=0.10) | **12.91°C** ✅ | 12.62°C |
| FireProt OOD PCC | ~0 | **0.44** ✅ | 0.49 |
| FireProt OOD R² | N/A | **-0.22** | -0.18 |
| Config | norm=False, mixup=0.0 | same | norm=False, mixup=0.0 |

---

## ✅ RESOLVED — ESM-2 Embedding Layer Mismatch (was 🔴 BLOCKING)

### Root Cause: Training used Layer 30, curation used Layer 36

**Diagnosis:** Computed cosine similarity between stored training embeddings and all 37 ESM-2 layers:
```
Layer 30: cos_sim=1.000000, norm=876.79, ratio=1.00  ← EXACT MATCH
Layer 36: cos_sim=0.979102, norm=13.15,  ratio=66.67  ← was being used
```

**Why:** ESM-2's final layer (36) applies LayerNorm → norms collapse to ~12. Layer 30 (pre-LayerNorm) preserves raw magnitude ~900 matching training data.

**Fix applied:**
- `curate_fireprot_holdout.py`: `REPR_LAYER = 36` → `REPR_LAYER = 30`
- `generate_esm2_embeddings.py`: default `--layers [36]` → `[30]`
- Verified: Training norm=939.85, Holdout norm=935.33, **ratio=1.005** ✓

### ✅ RESOLVED — Homology Filtering (was 🔴 BLOCKING)

Replaced `difflib.SequenceMatcher` with **CD-HIT-2D** at 40% identity threshold. Holdout set: 426 → **324 targets** (stricter filtering removed 102 homologous sequences).

### ✅ RESOLVED — TemBERTure/ESMStabP Real Inference (was 🔴 BLOCKING)

Real inference scripts written and executed. Predictions saved to `new_data/baseline_predictions.pt`.

---

## 🟡 NEEDS RE-RUN — Baseline Predictions Stale

Baseline predictions (TemBERTure/ESMStabP) were generated on the **old 426-target holdout**. New holdout has **324 targets** after CD-HIT filtering. No sequence keys stored in old predictions → can't align.

**Run this (~5-10 min, needs GPU):**
```bash
source /home/bibhu/miniconda3/etc/profile.d/conda.sh && conda activate stableprot_v2
cd /home/bibhu/Documents/temstampto/experiments/analysis
python run_baselines_inference.py
```
Script now also saves sequences for future alignment.

Then re-evaluate:
```bash
python evaluate_fireprot_generalization.py
```

---

## 🟡 REMAINING WORK

### BUG: V6 evaluate.py uses stale `best_model.pth`
**File:** `experiments/v6_multihead_esm2/evaluate.py`  
Points to `results/best_model.pth` but training saves per-seed to `results/seed{N}/model.pt`.

### No OGT Head Evaluation for V6
V6's `prepared_data_v2.pt` has no `test_ogt` split. V6 needs ESM-2 embeddings for 210K OGT test sequences (~50 hours GPU).

**Workaround for paper:** Use V5's OGT head results (5.77°C MAE). Same architecture, different embedding backbone.

### Statistical Tests
- Wilcoxon signed-rank test (V5 vs V6, V5 vs baselines)
- Paired t-test for MAE differences
- Not yet implemented

### SHAP Analysis
- Feature importance attribution for V5/V6 multi-head models
- Not yet implemented

---

## 📊 SOTA ASSESSMENT

### Current Published SOTA (2024-2025)

| Model | Dataset | MAE (°C) | PCC | Notes |
|-------|---------|----------|-----|-------|
| ESM3-DTm | s571 | 5.21 | 0.50 | Sequence + structure input |
| DeepSTABp | Internal test | 3.22 | — | Uses OGT + experimental conditions |
| TemBERTure | — | ~8-13 | ~0.33 | Sequence only, regression |
| ESMStabP | — | ~6-15 | ~0.30 | ESM-2 + Random Forest |

### StableProt V6 Results

| Dataset | MAE (°C) | PCC | R² |
|---------|----------|-----|----|
| ProThermDB (in-dist) | **5.40** | — | — |
| FireProt OOD (<40% ID) | **12.91** | 0.44 | -0.22 |

### Verdict: **Competitive, not yet SOTA**

**Strengths:**
- ProThermDB MAE 5.40°C → near ESM3-DTm (5.21) without needing structure
- **Dual-task** (OGT + Tm) — unique selling point vs single-task models
- Honest OOD evaluation with strict CD-HIT filtering

**Gaps to SOTA:**
- OOD MAE 12.91°C — all models struggle here, but room for improvement
- Negative R² on OOD → model explains variance poorly on unseen proteins
- DeepSTABp's 3.22°C is on internal test (easier), not comparable
- Missing structure modality (ESM3-DTm uses structures)

**Path to SOTA:**
1. Add structure features (ESMFold/AlphaFold embeddings)
2. Fine-tune ESM-2 backbone (currently frozen feature extraction)
3. Train on larger/more diverse datasets
4. Ensemble V5 (ProtT5) + V6 (ESM-2) predictions

---

## ✅ COMPLETED FIXES (All Sessions)

| Fix | What |
|-----|------|
| **ESM-2 layer (ROOT CAUSE)** | **REPR_LAYER 36 → 30 (Layer 30 = pre-LayerNorm, matches training norms ~940)** |
| Homology filtering | SequenceMatcher → CD-HIT-2D at 40% identity |
| Baseline inference | Real TemBERTure/ESMStabP predictions (not synthetic) |
| V6 config | target_normalization=False, mixup_alpha=0.0 |
| V6 retrain | ProThermDB MAE 5.75 → **5.40** |
| Dead code | Removed make_synthetic_baseline + make_ood_baseline |
| Eval denorm | Made V6 de-normalization conditional on config |
| Stale cache | Curation now force-regenerates ALL ESM-2 embeddings |
| File rename | reseults.md → results.md |
| Multi-threshold ROC | Added 45/55/65/80°C AUC + Global AUC to compute_metrics |
| Placeholder CSV | Fixed 'sequence' placeholder column → proper 'index' |
| ProtT5 tokenizer | AutoTokenizer → T5Tokenizer (Unigram crash fix) |
| Baseline alignment | Evaluation handles size mismatch + saves sequences |
