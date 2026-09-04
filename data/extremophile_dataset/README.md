# StableProt Extremophile Dataset Suite (OGT & Tm)

A curated, multi-regime benchmark and training suite specifically designed to evaluate and enhance protein language models on **extreme thermal regimes** (psychrophiles, moderate thermophiles, extreme thermophiles, and hyperthermophiles).

---

## 1. Thermal Regime Breakdown

### Optimal Growth Temperature (OGT) Extremophiles
Total Records: **34,293** unique proteins

| Thermal Regime | Range (°C) | Protein Count | Description |
|:---|:---:|:---:|:---|
| **Psychrophilic** | < 20.0 | 6,795 | Cold-adapted organisms (deep sea, polar) |
| **Moderate Thermophilic** | 50.0 – 65.0 | 91 | Industrial thermotolerant species |
| **Extreme Thermophilic** | 65.0 – 80.0 | 10,090 | Hot springs, hydrothermal vents (*Thermus*) |
| **Hyperthermophilic** | > 80.0 | 17,317 | Extreme hyperthermophilic Archaea (*Pyrococcus*, *Sulfolobus*) |

### Melting Temperature (Tm) Extremophiles
Total Records: **11,472** unique proteins

| Thermal Regime | Range (°C) | Protein Count | Description |
|:---|:---:|:---:|:---|
| **Psychro-labile / Low Tm** | < 45.0 | 7,776 | Highly flexible / heat-sensitive unfolding |
| **Moderate Thermophilic** | 65.0 – 75.0 | 1,464 | Thermostable biocatalysts |
| **Extreme Thermophilic** | 75.0 – 85.0 | 1,253 | Highly stable compact folds |
| **Hyperthermophilic** | ≥ 85.0 (up to 105°C) | 979 | Extreme thermal unfolding resistance |

### Paired OGT–Tm Cohort
Total Co-Annotated Proteins: **12,756** records with verified host environmental OGT and experimental unfolding Tm.

---

## 2. Directory Structure

```
data/extremophile_dataset/
├── README.md                                 # This specification
├── csv/
│   ├── extremophile_master_catalog.csv       # Unified catalog across all sources
│   ├── extremophile_ogt_benchmark.csv        # Dedicated OGT extremophile dataset
│   ├── extremophile_tm_benchmark.csv         # Dedicated Tm extremophile dataset
│   └── extremophile_paired_ogt_tm.csv        # Co-annotated OGT and Tm proteins
├── fasta/
│   ├── extremophile_ogt.fasta                # All OGT extremophile sequences
│   ├── extremophile_tm.fasta                 # All Tm extremophile sequences
│   ├── extremophile_hyperthermo.fasta        # Hyperthermophiles (>80°C / >=85°C)
│   └── extremophile_psychro.fasta            # Psychrophiles (<20°C / <45°C)
└── pt/
    ├── extremophile_benchmark.pt             # PyTorch bundle with cached SaProt embeddings & baseline predictions
    └── extremophile_train_val.pt             # PyTorch tensors for model fine-tuning
```

---

## 3. Usage Examples

### Loading with Pandas (Python)
```python
import pandas as pd

# Load OGT extremophiles
df_ogt = pd.read_csv("data/extremophile_dataset/csv/extremophile_ogt_benchmark.csv")
print("Hyperthermophiles count:", len(df_ogt[df_ogt['thermal_regime'] == 'hyperthermophile']))

# Load Tm extremophiles
df_tm = pd.read_csv("data/extremophile_dataset/csv/extremophile_tm_benchmark.csv")
print("Extreme thermostable Tm count:", len(df_tm[df_tm['target_temperature'] >= 75.0]))
```

### Loading PyTorch Benchmark Bundle
```python
import torch

data = torch.load("data/extremophile_dataset/pt/extremophile_benchmark.pt", map_location="cpu", weights_only=False)
ogt_bench = data['ogt_benchmark']
tm_bench = data['tm_benchmark']

print(f"Loaded {len(ogt_bench['records'])} OGT and {len(tm_bench['records'])} Tm benchmark records.")
```

---

## 4. Benchmark Evaluation
To evaluate StableProt vs literature baselines (TemBERTure, DeepSTABp, ESMStabP, ThermoFormer, PRIME) on this extremophile dataset, run:

```bash
/home/bibhu/miniconda3/envs/stableprot_v2/bin/python experiments/src/eval/evaluate_extremophile_benchmark.py
```
