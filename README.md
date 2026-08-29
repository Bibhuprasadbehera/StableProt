<p align="center">
  <img src="paper/writeup/figures/logos/flame_with_text.png" alt="StableProt Logo" width="460">
</p>

<p align="center">
  <b>Structure-Aware Protein Thermostability ($T_m$) & Environmental Adaptation ($OGT$) AI</b>
</p>

<p align="center">
  <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat-square&logo=pytorch" alt="PyTorch"></a>
  <a href="https://github.com/westlake-repl/SaProt"><img src="https://img.shields.io/badge/SaProt-650M%20Transformer-0074B8.svg?style=flat-square" alt="SaProt"></a>
  <a href="https://fastapi.tiangolo.com"><img src="https://img.shields.io/badge/FastAPI-Web%20Application-009688.svg?style=flat-square&logo=fastapi" alt="FastAPI"></a>
  <a href="#benchmarks"><img src="https://img.shields.io/badge/Evaluation-Strict%20Homology%20Audit%20(<30%25)-EAAC08.svg?style=flat-square" alt="Audit"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/License-Academic%20Research-0F172A.svg?style=flat-square" alt="License"></a>
</p>

---

## 🔬 Graphical Abstract

<p align="center">
  <img src="paper/writeup/figures/graphical_abstracts/stableprot_graphical_abstract_definitive.png" alt="StableProt Graphical Abstract" width="920">
</p>

---

## 🌟 Key Highlights

1. **Dual-Track 3Di & Sequence Tokenization**: Fuses primary amino acid sequences with Foldseek 3Di geometric structural tokens, giving the foundation model full structure-aware spatial intelligence directly from sequence.
2. **Decoupled Thermodynamic & Ecological Heads**: Disjoint neural pathways separate intrinsic protein thermal denaturation ($T_m$) from host organism optimum growth temperature ($OGT$), transferring evolutionary environmental priors ($\hat{y}_{\text{OGT}} \to T_m$).
3. **Calibrated Predictive Confidence Intervals**: Replaces single-point estimates with Gaussian probabilistic density profiles ($\mu \pm \sigma$), giving practitioners trustworthy uncertainty bounds.
4. **Interactive Web Application & In-Silico Loop Engineering**: Integrated interactive suite for single-sequence analysis, Chou–Fasman secondary structure & loop identification, and live mutant thermostability scoring.

---

## 📊 Benchmark Performance Summary

All evaluations use strict **bidirectional homology audits ($<30\%$ sequence identity)** between training and test sets to prevent data leakage.

| Benchmark Dataset | Metric | TemBERTure | ESMStabP | DeepSTABp | ThermoFormer | **StableProt (Ours)** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **ProThermDB (In-Distribution, n=3,340)** | MAE (°C) | 5.76 | 6.54 | 7.12 | 6.89 | **6.16** |
| | CRPS (°C) | 5.76 | 6.54 | 7.12 | 6.89 | **4.52** |
| **FireProtDB (OOD Holdout, n=322, <30% id)** | MAE (°C) | 12.76 | 13.12 | 14.05 | 13.84 | **11.85** |
| | CRPS (°C) | 12.76 | 13.12 | 14.05 | 13.84 | **8.71** |
| **Bin-Balanced OGT (Brenda/BacDive, n=1,200)** | MAE (°C) | 9.42 | 9.80 | 10.15 | 8.94 | **6.48** |
| **Prospective Marine Carrageenases (n=13)** | Acc (%) | 61.5% | 53.8% | 46.2% | 61.5% | **81.8%** |

---

## 🚀 Quickstart & Web Interface

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Bibhuprasadbehera/StableProt.git
cd StableProt

# Create and activate conda environment
conda create -n stableprot python=3.10 -y
conda activate stableprot

# Install dependencies
pip install -r requirements.txt
```

### 2. Launching the Interactive Web Suite

```bash
# Start the FastAPI + Jinja2 web application on port 8000
python -m uvicorn inference.main:app --host 0.0.0.0 --port 8000
```
Open **`http://localhost:8000`** in your browser to access:
- **Home**: Architecture overview and benchmark highlights.
- **Predict**: Instant $T_m$ and $OGT$ predictions with calibrated thermometers and amino acid composition analysis.
- **Design**: Chou–Fasman loop detection with live in-silico mutation testing and trial history logging.
- **About**: Model details, terms of use, and citation info.

---

## 💻 Programmatic Python API

```python
from inference.v9_predict import V9Predictor

# Initialize the 5-seed ensemble predictor
predictor = V9Predictor(models_dir="experiments/src/training/v9_disjoint/results")

# Predict on a protein sequence
sequence = "RPDFCLEPPYTGPCKARIIRYFYNAKAGLCQTFVYGGCRAKRNNFKSAEDCMRTCGGA"
result = predictor.predict_single(sequence)

print(f"Melting Temperature (Tm): {result['tm_pred']:.2f} ± {result['tm_conf']:.2f} °C")
print(f"Optimal Growth Temp (OGT): {result['ogt_pred']:.2f} ± {result['ogt_conf']:.2f} °C")
print(f"Thermal Tier: {result.get('thermal_tier', 'Mesophilic')}")
```

---

## 📖 Citation

If you find StableProt useful in your research, please cite our manuscript:

```bibtex
@article{behera2026stableprot,
  title={StableProt: Structure-Aware Deep Learning for Protein Thermostability ($T_m$) and Environmental Adaptation ($OGT$) Prediction with Calibrated Confidence Intervals},
  author={Behera, Bibhu Prasad and Daxit, Anshuman},
  journal={Working Manuscript},
  year={2026},
  publisher={iBRIC--Institute of Life Sciences},
  url={https://github.com/Bibhuprasadbehera/StableProt}
}
```

---

## 📬 Contact & Affiliation

**Computational Biology and Bioinformatics Laboratory**  
*iBRIC–Institute of Life Sciences (ILS), Bhubaneswar, Odisha, India*  
- **Bibhu Prasad Behera**: `bibhu.prasad@ils.res.in`  
- **Dr. Anshuman Dixit**: `anshuman@ils.res.in`