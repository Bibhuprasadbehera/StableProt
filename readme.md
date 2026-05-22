# TemStaPro Development Log
---
## 2026-04-28 — Environment Setup Complete
### Component Status
| Component | Status |
|-----------|--------|
| Conda environment `temstapro_env_CPU` | Created |
| Python version | 3.7.15 |
| PyTorch | 1.13.0 (CPU) |
| NumPy | 1.21.6 |
| Transformers | 4.24.0 |
| Tests (`make all`) | All 14 tests passed |
### How to Use
**Activate the environment:**
```bash
source /home/bibhu/miniconda3/etc/profile.d/conda.sh && conda activate temstapro_env_CPU
```
**Run TemStaPro:**
```bash
cd TemStaPro
./temstapro -f <FASTA_FILE> -d <PROTTRANS_MODEL_DIR> --mean-output <OUTPUT.tsv>
```
**Example:**
```bash
cd TemStaPro
./temstapro -f tests/data/short_sequence.fasta -d ./ProtTrans/ -e ./tests/outputs/ --mean-output ./predictions.tsv
```
**Deactivate when done:**
```bash
conda deactivate
```
---
## 2026-04-28 — Codebase Analysis
### Architecture Overview
TemStaPro is a Python tool for predicting protein thermostability from amino acid sequences using embeddings from protein language models (ProtTrans/ProtT5-XL). The codebase consists of 6 main files:
| File | Purpose |
|------|---------|
| `temstapro` | Main entry point — CLI argument parsing, orchestration |
| `prottrans_models.py` | ProtTrans model loading, FASTA parsing, embedding generation |
| `data_process.py` | Embedding collection, caching, dataset construction |
| `results.py` | Inference interpretation, TSV output, plot generation |
| `model_flow.py` | ML model training and prediction pipeline |
| `README.md` | Documentation |
### Data Flow
```
FASTA Input
    → process_FASTA()
    → get_sequences_without_embeddings()
    → get_embeddings()
    → ProtT5-XL model
    → mean / per-res embeddings
    → collect_mean_embeddings() / collect_per_res_embeddings()
    → predict_with_models()
    → get_temperature_label()
    → TSV output
    → plot_per_res_inferences()
    → SVG plots
```
### Key Functions
#### `temstapro` (Main CLI)
| Option | Description |
|--------|-------------|
| `-f` | FASTA input (required) |
| `-d` | ProtTrans model directory (required) |
| `-e` | Embeddings cache directory (optional) |
| `--mean-output` | Mean prediction TSV output (optional) |
| `--per-res-output` | Per-residue prediction TSV output (optional) |
| `--per-segment-output` | Per-segment prediction TSV output (optional) |
| `-p` | Plot output directory (optional) |
| `--more-thresholds` | Enable t20 classifier |
| `--portion-size` | Batch processing control |
#### `process_FASTA()` (`prottrans_models.py:73`)
- Parses FASTA, replaces U/Z/O with X
- Returns `(seqs, orig_seq_headers, orig_seqs)` dict
#### `get_embeddings()` (`prottrans_models.py:102`)
- Batch processing with `max_residues=4000`, `max_seq_len=2000`, `max_batch=100`
- Handles OOM by falling back to single-sequence processing
- Returns `{'per_res_representations': {}, 'mean_representations': {}}`
#### `collect_mean_embeddings()` (`data_process.py:30`)
- Loads from cache or fresh embeddings
- Creates `x_test` (embedding tensors) and `y_test` (label=999 placeholder)
#### `collect_per_res_embeddings()` (`data_process.py:63`)
- Supports averaging smoothing with `WINDOW_SIZE` parameter
- Creates per-residue dataset with position tracking
#### `predict_with_models()` (`model_flow.py:181`)
- Loads trained models from `models/` directory
- Averages predictions across thresholds
- Returns `averaged_inferences`, `binary_inferences`, `labels`, `clashes`
#### `get_temperature_label()` (`results.py:8`)
- Left-hand: finds first threshold where prediction < 0.5
- Right-hand: finds last threshold where prediction >= 0.5
#### `detect_clash()` (`results.py:37`)
- Detects inconsistent predictions in ensemble
- Returns `*` for clash, `-` for consistent
### Test Cases Summary
| Test | Description |
|------|-------------|
| `001-002` | Basic mean predictions with cache |
| `003` | Per-residue predictions |
| `004` | Multiple sequences, no cache |
| `005` | Per-segment predictions |
| `006` | Multiple sequences with per-residue |
| `007` | Multiple short sequences with plotting |
| `008` | Multiple sequences, no cache |
| `009` | Long sequence, no cache |
| `010` | Multiple sequences with plotting and temp files |
| `011` | Long sequence per-residue |
| `012` | Multiple sequences per-residue |
| `013` | Cache loading test (two runs) |
| `014` | t20 classifier with `--more-thresholds` |
### Key Code Patterns
1. **Embedding caching**: Uses SHA256 hash of sequence as filename suffix: `mean_{sha256(seq)}.pt`
2. **Data format**: Embeddings stored as PyTorch tensors with `{"label": seq_id, "sequence": str, "mean_representations": tensor}`
3. **Model loading**: Checks for local `pytorch_model.bin` first, then downloads from server path
4. **Batch processing**: Sorts sequences by length (descending) for efficient batching

---
## 2026-04-30 — Experimentation Framework (V0, V1, V2)

We have established a structured framework in the `experiments/` directory to compare different training strategies and models.

### 1. Data Preparation

Before training or evaluation, data must be parsed and embeddings generated.
- **Script**: `experiments/prepare_data.py`
- **Full Data**: Generates embeddings for all ~1M sequences in the dataset.
  ```bash
  python prepare_data.py --full --output prepared_data_full.pt
  ```
- **Caching**: Individual embeddings are cached in `experiments/embeddings_cache/` to allow resumption and reuse across experiments.

### 2. Version Details

| Version | Name | Key Features | Purpose |
|---------|------|--------------|---------|
| **V0** | **Original** | Pre-trained TemStaPro models (40-80°C) | Gold standard benchmark. |
| **V1** | **Baseline** | MLP, BCELoss, Thresholds 5-95°C | Baseline for local retraining on full range. |
| **V2** | **Improved** | Weighted Loss, Balanced Sampling, Dropout, BatchNorm, 5-95°C | Advanced training for class imbalance. |
| **V3** | **Regression** | MLP (outputs raw temp), MSE Loss | Direct continuous OGT prediction. |
| **V4** | **Improved Regression** | Huber loss, residual conn, LR scheduler, mixup, target norm | Better training for regression. |
| **V5** | **Multi-Head ProtT5** | Shared backbone + OGT/Tm heads (ProtT5 1024-dim) | Multi-task learning. |
| **V6** | **Multi-Head ESM-2** | Shared backbone + OGT/Tm heads (ESM-2 2560-dim) | ESM-2 upgrade over V5. |

### 3. How to Run & Compare

Each version is contained in its own subdirectory.

#### Evaluate / Train
```bash
cd experiments/v0_original && python evaluate.py
cd experiments/v1_baseline && python train.py
cd experiments/v2_improved && python train.py
cd experiments/v3_regression && python train.py
cd experiments/v4_improved && python train.py
cd experiments/v5_multihead && python train.py
cd experiments/v6_multihead_esm2 && python train.py
```

#### Unified Comparison
```bash
cd experiments
python compare_results.py
```
This script unifies all models by converting binary outputs (V0, V1, V2) into continuous OGT estimates (Expected Value) to compare directly against V3/V4 Regression.

### 4. Key Metrics for Comparison

Comparison focuses on continuous OGT accuracy across the entire 0-100°C range:
- **MAE / RMSE**: Overall error in degrees Celsius.
- **Spearman ρ**: Rank correlation (biologically relevant).
- **Per-Bin MAE**: Error broken down by extremity (Psychrophiles <20°C, Mesophiles 20-40°C, Thermophiles 40-60°C, Extreme 60-80°C, Hyperthermophiles >80°C).

### 5. Current Findings
- Training on the **Full Dataset** (900k sequences) is essential. GPU acceleration cuts embedding generation from days to ~12 hours.
- Binary classifiers naturally struggle at extreme bounds. Regression (V3) aims to solve this.
- V4 improvements (Huber, residual, mixup) give marginal gains on OGT data (MAE 5.75 vs V3's 5.66) — OGT labels are clean enough that MSE suffices.
- V2 (20 specialized binary models) dominates at temperature extremes but cannot produce continuous predictions.
- WeightedRandomSampler hurts overall MAE on OGT data — removed from V4.
- Per-sample bin weighting needs careful tuning; naive application degrades performance.

---
## 2026-05-13 — V4/V5/V6 Split & Multi-Head Architecture

### Key Decisions
1. **Renamed** old v4_multihead → v5_multihead (ProtT5) and v5 → v6_multihead_esm2 (ESM-2)
2. **Created V4** as improved single-head regression (V3 + training tricks)
3. **Generated ProtT5 embeddings** for Tm datasets (Meltome 24K + ProThermDB 5.5K + TemBERTure 531)
4. **Test set alignment**: V0-V4 all evaluated on same 210K OGT test set

### V4 Results (Final Retraining)
- Ensemble MAE: 5.724°C on 210K OGT test set (improved over initial V4's 5.746°C, close to V3's 5.664°C).
- Architecture: Single-head MLP with residual connection, MSE loss, cosine annealing LR scheduler, and gradient clipping.
- Finding: Dropping target normalization and Huber loss restored optimal convergence on clean OGT targets.

### V5 Multi-Head ProtT5 Completion
- Unified Data Prep: Combined 940K OGT sequences and ~24K Meltome Tm sequences with distinct evaluation sets.
- Fixed Training Bugs: Addressed a CSV/FASTA ID parsing mismatch and filtered out two corrupted NaN embedding rows that caused gradient explosions (`OGT=nan`).
- Target normalization and mixup augmentation were disabled to prevent target scale shifting during alternating multi-head updates.
- Performance: OGT Head ensemble MAE of 5.774°C on the 210K test set. Tm Head ensemble MAE of 7.290°C on the ProThermDB test set. Highly stable convergence achieved across all 5 random seeds.