# TemStaPro Experiments — Help Guide

Complete guide for running the experiment pipeline: data preparation, training, evaluation, comparison, and inference.

---

## Quick Start

```bash
# 1. Activate environment
source /home/bibhu/miniconda3/etc/profile.d/conda.sh && conda activate temstapro_env_CPU

# 2. Prepare data (generates ProtT5 embeddings — takes ~30-60 min on CPU with default sample)
cd /home/bibhu/Documents/temstampto/experiments
python prepare_data.py --train-sample 5000 --val-sample 1000 --test-sample 2000

# 3. Evaluate V0 Original (pre-trained models, no retraining)
cd v0_original
python evaluate.py

# 4. Train V1 Baseline
cd ../v1_baseline
python train.py

# 5. Train V2 Improved
cd ../v2_improved
python train.py

# 6. Compare ALL results
cd ..
python compare_results.py
```

---

## Directory Structure

```
experiments/
├── common/                    # Shared code
│   ├── data_utils.py          # FASTA parsing, embedding generation, dataset creation
│   └── metrics.py             # AUC-ROC, F1, MCC, plotting functions
├── v0_original/               # Original pre-trained models (NO retraining)
│   ├── evaluate.py            # Loads StableProt/models/ and evaluates
│   └── results/               # [generated] Metrics, predictions, plots
├── v1_baseline/               # Retrained: standard BCE, no regularization
│   ├── model.py               # MLP_Baseline (original architecture)
│   ├── config.py              # Hyperparameters
│   ├── train.py               # Training script
│   └── results/               # [generated] Checkpoints, metrics, plots
├── v2_improved/               # Builds on V1 + weighted loss + balanced sampling + dropout
│   ├── model.py               # MLP_Improved (adds BatchNorm + Dropout to V1)
│   ├── config.py              # Hyperparameters
│   ├── train.py               # Training script
│   └── results/               # [generated] Checkpoints, metrics, plots
├── comparison/                # [generated] Multi-experiment comparison plots
├── embeddings_cache/          # [generated] Cached ProtT5 embeddings
├── prepared_data.pt           # [generated] Preprocessed data
├── prepare_data.py            # Data preparation script
├── compare_results.py         # Results comparison script (auto-discovers all vX folders)
├── suggestions.md             # Future improvement ideas
└── help.md                    # This file
```

### Experiment Progression (incremental improvements)

```
V0 Original       → Pre-trained models as-is (baseline benchmark)
    ↓
V1 Baseline        → Same architecture, retrained on your data split (BCELoss, no regularization)
    ↓  (adds: weighted loss + balanced sampling + dropout + batchnorm + weight decay)
V2 Improved        → Everything from V1 + all improvements
```

---

## Step 1: Data Preparation

The `prepare_data.py` script parses FASTA files, samples sequences, generates ProtT5 embeddings, and saves everything as a single `.pt` file.

### Basic Usage

```bash
cd /home/bibhu/Documents/temstampto/experiments

# Default: 5000 train, 1000 val, 2000 test (recommended for first experiment)
python prepare_data.py

# Custom sample sizes
python prepare_data.py --train-sample 10000 --val-sample 2000 --test-sample 3000

# Use all data (WARNING: very slow on CPU — days for 943K sequences)
python prepare_data.py --full

# Custom output path
python prepare_data.py --output /path/to/my_data.pt
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--train-sample` | 5000 | Number of training sequences to sample |
| `--val-sample` | 1000 | Number of validation sequences to sample |
| `--test-sample` | 2000 | Number of test sequences to sample |
| `--full` | False | Use all data (no sampling) |
| `--seed` | 42 | Random seed for reproducible sampling |
| `--output` | `prepared_data.pt` | Output file path |
| `--cache-dir` | `embeddings_cache/` | Directory for cached embeddings |
| `--model-dir` | `StableProt/ProtTrans/` | ProtT5 model directory |
| `--max-seq-len` | 1500 | Filter out sequences longer than this |

### Time Estimates (CPU)

| Sample Size | Approximate Time |
|------------|-----------------|
| 1000 | ~10-15 min |
| 5000 | ~45-60 min |
| 10000 | ~1.5-2 hrs |
| 50000 | ~8-12 hrs |
| Full (943K) | ~3-5 days |

> **Tip**: Embeddings are cached! If you re-run with a larger sample, previously computed embeddings will be reused automatically.

---

## Step 2: Evaluate & Train

### V0 Original (Pre-trained models — evaluation only)

```bash
cd /home/bibhu/Documents/temstampto/experiments/v0_original

# Evaluate all thresholds (40, 45, 50, 55, 60, 65°C)
python evaluate.py

# Evaluate specific thresholds only
python evaluate.py --thresholds 40 65
```

**V0 is:**
- The original pre-trained models from `StableProt/models/`
- NO retraining — just loads and evaluates
- Uses the original `MLP_C2H2` class with PyTorch Lightning state dict remapping
- This is your true baseline benchmark

### V1 Baseline (Retrained, Standard Binary Classification)

```bash
cd /home/bibhu/Documents/temstampto/experiments/v1_baseline

# Train all thresholds (40, 45, 50, 55, 60, 65°C)
python train.py

# Train specific thresholds only
python train.py --thresholds 40 65

# Override data path or epochs
python train.py --data /path/to/prepared_data.pt --epochs 30
```

**V1 uses:**
- `BCELoss` (unweighted)
- Standard random batching (no balanced sampling)
- No Dropout, no BatchNorm
- Early stopping with patience=10

### V2 Improved (Weighted + Balanced + Regularized)

```bash
cd /home/bibhu/Documents/temstampto/experiments/v2_improved

# Train all thresholds
python train.py

# Train specific thresholds only
python train.py --thresholds 40 65

# Override data path or epochs
python train.py --data /path/to/prepared_data.pt --epochs 30
```

**V2 uses:**
- `BCEWithLogitsLoss` with auto `pos_weight` (upweights minority class)
- `WeightedRandomSampler` (balanced batches)
- Dropout (0.3 → 0.2) + BatchNorm
- L2 weight decay (1e-5)
- Early stopping with patience=10

### Training Options

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `../prepared_data.pt` | Path to preprocessed data |
| `--thresholds` | `[40,45,50,55,60,65]` | Temperature thresholds to train |
| `--epochs` | 50 | Override number of training epochs |

### What Each Training Run Outputs

```
results/
├── t40/                    # Results for threshold 40°C
│   ├── seed1/
│   │   ├── model.pt        # Saved model weights
│   │   ├── metrics.json    # All evaluation metrics
│   │   ├── predictions.pt  # Raw predictions (y_true, y_prob)
│   │   ├── roc_curve.png   # ROC curve plot
│   │   └── training_history.png  # Loss/AUC over epochs
│   ├── seed2/ ...
│   ├── seed3/ ...
│   ├── seed4/ ...
│   ├── seed5/ ...
│   └── ensemble/           # Averaged predictions across all seeds
│       ├── metrics.json
│       ├── predictions.pt
│       ├── roc_curve.png
│       └── prc_curve.png
├── t45/ ...
├── t50/ ...
└── summary.json            # All threshold results in one file
```

---

## Step 3: Compare Results

The comparison script **auto-discovers** all `vX_*` folders that have a `results/` directory.

```bash
cd /home/bibhu/Documents/temstampto/experiments

# Compare ALL experiments (auto-discovers v0, v1, v2, and any future vN)
python compare_results.py

# Compare specific thresholds only
python compare_results.py --thresholds 40 65

# Compare specific experiments only
python compare_results.py --experiments v0_original v2_improved
```

### What It Generates

```
comparison/
├── roc_comparison_t40.png       # Multi-experiment ROC curves for 40°C
├── roc_comparison_t45.png       # ... for 45°C
├── ...
├── metrics_comparison_t40.png   # Bar chart: all experiments' key metrics
├── metrics_comparison_t45.png   # ...
└── comparison.json              # Full comparison data
```

### Console Output Includes

- Per-threshold metric table with all experiments side-by-side
- Delta (Δ) between first and last experiment with ↑/↓ arrows
- Grand summary tables for AUC-ROC and MCC across all thresholds
- Best score indicator per threshold

---

## Step 4: Inference with Trained Models

To use a trained model for inference on new sequences:

```python
import torch
import sys, os

# Add experiments to path
sys.path.insert(0, '/home/bibhu/Documents/temstampto/experiments')

from common.data_utils import parse_fasta_with_temps, generate_embeddings

# For V1 models:
from v1_baseline.model import MLP_Baseline
model = MLP_Baseline()
state = torch.load('v1_baseline/results/t65/seed1/model.pt')
model.load_state_dict(state)
model.eval()

# For V2 models:
from v2_improved.model import MLP_Improved
model = MLP_Improved()
state = torch.load('v2_improved/results/t65/seed1/model.pt')
model.load_state_dict(state)
model.eval()

# Generate embeddings for new sequences
records = parse_fasta_with_temps('your_sequences.fasta')
embeddings = generate_embeddings(records)

# Get predictions
with torch.no_grad():
    probs = model.predict_proba(embeddings)
    for i, (seq_id, _, _) in enumerate(records):
        print(f"{seq_id}: {probs[i].item():.4f} (thermophilic)" if probs[i] > 0.5
              else f"{seq_id}: {probs[i].item():.4f} (mesophilic)")
```

---

## Evaluation Metrics Explained

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| **AUC-ROC** | Area under ROC curve | Overall discriminative ability (threshold-independent) |
| **AUC-PRC** | Area under Precision-Recall curve | Better than AUC-ROC for imbalanced data |
| **F1** | Harmonic mean of precision & recall | Balances false positives and false negatives |
| **MCC** | Matthews Correlation Coefficient | Best single metric for imbalanced classification |
| **Balanced Accuracy** | (Sensitivity + Specificity) / 2 | Not fooled by class imbalance |
| **Sensitivity** | TP / (TP + FN) | "How many thermophiles did we catch?" |
| **Specificity** | TN / (TN + FP) | "How many mesophiles did we correctly exclude?" |
| **Precision** | TP / (TP + FP) | "When we predict thermophilic, how often are we right?" |

> **Important**: With imbalanced data, **accuracy is misleading**. A model predicting "mesophilic" for everything gets 93% accuracy at the 65°C threshold. Focus on **MCC**, **F1**, and **AUC-PRC** instead.

---

## Modifying Hyperparameters

Edit `config.py` in each experiment folder:

```python
# v1_baseline/config.py or v2_improved/config.py
CONFIG = {
    'learning_rate': 1e-3,       # Try: 5e-4, 1e-4
    'batch_size': 64,            # Try: 32, 128
    'num_epochs': 50,            # Increase if not converging
    'early_stopping_patience': 10,
    'hidden_size_1': 512,        # Try: 256, 1024
    'hidden_size_2': 256,        # Try: 128, 512
    'thresholds': [40, 45, 50, 55, 60, 65],  # Add: 70, 75, 80
    'seeds': [1, 2, 3, 4, 5],   # More seeds = more stable ensemble
    # V2 only:
    'dropout_1': 0.3,            # Try: 0.1, 0.5
    'dropout_2': 0.2,            # Try: 0.1, 0.3
    'pos_weight_cap': 50.0,      # Cap for extreme weight values
}
```

---

## Adding New Experiments

To create a new experiment version (e.g., v3 with Focal Loss):

1. Copy an existing folder: `cp -r v2_improved v3_focal_loss`
2. Modify `model.py` and/or `config.py`
3. Update loss function in `train.py`
4. Run: `cd v3_focal_loss && python train.py`
5. Compare: `python compare_results.py`  ← auto-discovers v3!

The `compare_results.py` script auto-discovers ANY folder starting with `v` that has a `results/` subdirectory. Just name your folder `v3_*`, `v4_*`, etc. and it will be picked up automatically.

See `suggestions.md` for specific strategies to try.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: sklearn` | `pip install scikit-learn==0.24.2` |
| `CUDA out of memory` | Reduce `--train-sample` or `batch_size` |
| `prepared_data.pt not found` | Run `python prepare_data.py` first |
| Slow embedding generation | Use `--train-sample 1000` for quick test, embeddings are cached for reuse |
| ProtT5 model not found | Check `StableProt/ProtTrans/pytorch_model.bin` exists |
| All predictions are 0 or 1 | Check class balance; try V2 with weighted loss |
