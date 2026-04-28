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