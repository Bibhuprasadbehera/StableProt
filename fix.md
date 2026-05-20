# StableProt V2 — Fix Tracker

> Last updated: 2026-05-20

## Current V6 Status

| Metric | Old V6 | Retrained V6 | V5 (ProtT5) |
|--------|--------|-------------|-------------|
| ProThermDB Tm MAE | 5.75°C | **5.40°C** ✅ | 7.29°C |
| FireProt OOD MAE | ??? | **TBD — stale cache** | 12.59°C |
| Config | norm=True, mixup=0.2 | norm=False, mixup=0.0 | norm=False, mixup=0.0 |

---

## 🔴 BLOCKING — Must Fix Before Any Results

### 1. FireProt ESM-2 embeddings are from WRONG LAYER (stale cache)
**Status:** Code fixed, needs re-run  
The curation script was checking cache before regenerating. All 426 FireProt sequences had cached embeddings from the old Layer 22 run — so the Layer 36 fix had NO EFFECT. The V6 FireProt results (MAE 14.95) are still on Layer 22 embeddings.

**Fix applied:** `curate_fireprot_holdout.py` now force-regenerates ALL ESM-2 embeddings (no cache skip).

**YOU NEED TO RUN (requires GPU, ~15 min):**
```bash
source /home/bibhu/miniconda3/etc/profile.d/conda.sh && conda activate stableprot_v2
cd /home/bibhu/Documents/temstampto/experiments/data_processing
python curate_fireprot_holdout.py

# Then re-evaluate:
cd /home/bibhu/Documents/temstampto/experiments/analysis
python evaluate_fireprot_generalization.py
```

### 2. TemBERTure/ESMStabP numbers — NOT from real inference
The numbers in `results.md` Tables 1 & 3 (TemBERTure MAE 8.35, ESMStabP MAE 6.42) have NO provenance. No inference scripts, no prediction files, no model checkpoints found in the project. These cannot appear in the paper.

**Needs:**
- Clone TemBERTure repo, run inference on ProThermDB + FireProt sequences
- Clone ESMStabP repo, run inference on same sequences
- Use per-sequence predictions for fair comparison

### 3. Homology filtering uses SequenceMatcher (not CD-HIT)
`curate_fireprot_holdout.py` filters at "30% identity" using Python's `difflib.SequenceMatcher`. This is NOT biological sequence identity. Reviewers will flag this.

**Fix:** Replace SequenceMatcher with CD-HIT at 30% threshold, or at minimum document this as a limitation and run CD-HIT validation.

---

## 🟡 REMAINING WORK

### BUG 4: V6 evaluate.py uses stale `best_model.pth`
**File:** `experiments/v6_multihead_esm2/evaluate.py`  
Points to `results/best_model.pth` but training saves per-seed to `results/seed{N}/model.pt`. The 6.3MB `best_model.pth` is from the OLD training run. Either delete it or update evaluate.py to use ensemble.

### No OGT Head Evaluation for V6
V6's `prepared_data_v2.pt` has no `test_ogt` split. V5's ProtT5 data does, but V6 needs ESM-2 embeddings for 210K OGT test sequences. These don't exist yet — generating them requires ~50 hours of GPU time.

**Workaround for paper:** Use V5's OGT head results (5.77°C MAE) to establish OGT baseline. V6 architecture is identical — report "same architecture, different embedding" and note V6 OGT eval as future work.

### `reseults.md` → `results.md` ✅ DONE
Renamed.

---

## 📋 ACTION ITEMS (Priority Order)

### You Run (GPU required):
1. **Re-run FireProt curation** — regenerates ESM-2 Layer 36 embeddings (~15 min)
2. **Re-run FireProt evaluation** — gets honest V6 OOD numbers (~1 min)
3. **Clone & run TemBERTure** on ProThermDB + FireProt sequences (~1 hour setup)
4. **Clone & run ESMStabP** on ProThermDB + FireProt sequences (~1 hour setup)

### I Do (no GPU):
5. ~~Multi-threshold ROC implementation~~ — IN PROGRESS
6. ~~TemBERTure/ESMStabP inference scripts~~ — TO DO
7. ~~Dual-task comparison table generator~~ — TO DO
8. ~~Statistical tests (Wilcoxon)~~ — TO DO
9. ~~Stratified error analysis~~ — TO DO

---

## 🔵 TESTING STRATEGY

### Dual-Task Evaluation (Paper Story)

| Model | OGT MAE | Tm MAE (ProThermDB) | Tm MAE (FireProt OOD) | Both? |
|-------|---------|--------------------|-----------------------|-------|
| TemStaPro V0 | 10.95 | 12.62 (proxy) | 20.86 | OGT only |
| TemBERTure | ❌ N/A | TBD (real inference) | TBD | Tm only |
| ESMStabP | ❌ N/A | TBD (real inference) | TBD | Tm only |
| **V6** | ~5.7* | **5.40** | **TBD** | **✅ Both** |

*Using V5 OGT head as proxy (same architecture)

### ROC Improvement ✅ DONE
Multi-threshold analysis at 45/55/65/80°C added to `compute_metrics()`. Also added Global AUC (mean ROC AUC across 30-90°C thresholds).

---

## ✅ COMPLETED FIXES (This Session)

| Fix | What |
|-----|------|
| ESM-2 layer | REPR_LAYER 22 → 36 in curate + eval scripts |
| V6 config | target_normalization=False, mixup_alpha=0.0 |
| V6 retrain | ✅ Done. ProThermDB MAE 5.75 → **5.40** |
| Dead code | Removed make_synthetic_baseline + make_ood_baseline |
| Eval denorm | Made V6 de-normalization conditional on config |
| Stale cache | Curation now force-regenerates ALL ESM-2 embeddings |
| File rename | reseults.md → results.md |
| Impl plan | Removed experiments B-F, added V6 retrain instructions |
| Multi-threshold ROC | Added 45/55/65/80°C AUC + Global AUC to compute_metrics |
| Placeholder CSV | Fixed 'sequence' placeholder column → proper 'index' |
| Stale comment | Updated "Layer 22" comments to "Layer 36" everywhere |

