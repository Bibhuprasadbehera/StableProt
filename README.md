<p align="center">
  <img src="assets/flame_with_text.png" alt="StableProt Logo" width="460">
</p>

<p align="center">
  <b>Structure-Aware Protein Thermostability (<i>T<sub>m</sub></i>) & Environmental Adaptation (<i>OGT</i>) Deep Learning</b>
</p>

<p align="center">
  <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat-square&logo=pytorch" alt="PyTorch"></a>
  <a href="https://github.com/westlake-repl/SaProt"><img src="https://img.shields.io/badge/SaProt-650M%20Transformer-0074B8.svg?style=flat-square" alt="SaProt"></a>
  <a href="https://fastapi.tiangolo.com"><img src="https://img.shields.io/badge/FastAPI-Web%20Application-009688.svg?style=flat-square&logo=fastapi" alt="FastAPI"></a>
  <a href="#benchmarks"><img src="https://img.shields.io/badge/Evaluation-Strict%20Homology%20Audit%20(<30%25)-EAAC08.svg?style=flat-square" alt="Audit"></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-MIT-0F172A.svg?style=flat-square" alt="License"></a>
</p>

---

## 🌟 Key Highlights

1. **Structure-Aware 3Di Tokenization**: Fuses primary amino acid sequences with Foldseek 3Di geometric structural states through a frozen SaProt 650M backbone, capturing cooperative tertiary packing and surface loop flexibility without high-overhead fine-tuning.
2. **Decoupled Multi-Head Architecture**: Parameter-isolated neural pathways eliminate destructive gradient interference ($\cos\theta = -0.077 \to 0$) between organismal growth limits ($OGT$) and single-molecule melting temperatures ($T_m$), feeding an environmental prior link ($\hat{y}_{\text{OGT}} \to T_m$).
3. **Calibrated 95% Predictive Intervals**: Replaces uncalibrated point estimates with observation-level heteroscedastic uncertainty ($\mu \pm 1.96 c^* \sigma$), evaluated via Continuous Ranked Probability Score (CRPS).
4. **Targeted Mesophilic Subsampling (14%)**: Dynamically counteracts extreme database imbalance, expanding thermophilic training representation from 16.8% to 38.0% and mitigating mesophilic prediction collapse on high-temperature biocatalysts ($T_m \ge 80^\circ\text{C}$).
5. **Interactive Web Suite & Loop Design Studio**: Production-ready FastAPI interface providing two-stage ensemble inference, dynamic stability gauges, secondary structure profiling, and in-silico mutagenesis for surface loop stabilization.

---

## 📊 Benchmark Performance Summary

All evaluations enforce strict **bidirectional homology decontamination ($<30\%$ sequence identity)** between training and evaluation partitions to ensure zero data leakage.

| Benchmark Dataset | Metric | TemBERTure | ThermoFormer-TM | DeepSTABp | ESMStabP | **StableProt (Calibrated)** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **ProThermDB (In-Distribution, $n=3{,}340$)** | MAE (°C) $\downarrow$ | 5.76 | 7.05 | 7.11 | 9.14 | **6.16** |
| | CRPS (°C) $\downarrow$ | 5.76 | 7.05 | 7.11 | 9.14 | **4.52** |
| | Pearson $r$ $\uparrow$ | 0.826 | 0.829 | 0.812 | 0.715 | **0.788** |
| | Recall ($T_m \ge 80^\circ\text{C}$) $\uparrow$ | 57.1% | 90.3% | 55.9% | 86.1% | **92.0%** |
| | Screening F1 $\uparrow$ | 0.568 | 0.618 | 0.535 | 0.617 | **0.650** |
| **FireProtDB (Zero-Shot OOD, $n=322$)** | MAE (°C) $\downarrow$ | 12.76 | 13.61 | 13.59 | 14.91 | **11.85** |
| | CRPS (°C) $\downarrow$ | 12.76 | 13.61 | 13.59 | 14.91 | **8.71** |
| | Spearman $\rho$ $\uparrow$ | 0.214 | 0.233 | 0.239 | 0.236 | **0.350** |
| | Recall ($T_m \ge 80^\circ\text{C}$) $\uparrow$ | 42.9% | 63.6% | 45.5% | 54.5% | **81.8%** |
| **Prospective Enzyme Screen ($N=87$)** | 95% CI Capture $\uparrow$ | — | — | — | — | **76.5%** (65/85) |

*Note: For deterministic baselines lacking predictive intervals, CRPS equals their exact MAE without mathematical penalty.*

---

## 🚀 Quickstart & Installation

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
- **Predict**: Instant $T_m$ and $OGT$ predictions with calibrated thermometers and confidence intervals.
- **Biophysical Profiling**: Secondary structure propensity and surface loop visualization.
- **Loop Design Studio**: In-silico mutagenesis for surface loop stabilization.

---

## 💻 Programmatic Python API

```python
from inference.v9_predict import V9Predictor

# Initialize the 5-seed production ensemble predictor
predictor = V9Predictor(models_dir="experiments/src/training/v9_disjoint/results")

# Predict on a candidate protein sequence
sequence = "RPDFCLEPPYTGPCKARIIRYFYNAKAGLCQTFVYGGCRAKRNNFKSAEDCMRTCGGA"
result = predictor.predict_single(sequence)

print(f"Melting Temperature (Tm): {result['tm_pred']:.2f} ± {result['tm_conf']:.2f} °C")
print(f"Optimal Growth Temp (OGT): {result['ogt_pred']:.2f} ± {result['ogt_conf']:.2f} °C")
print(f"Thermal Classification: {result.get('thermal_tier', 'Mesophilic')}")
```

---

## 📖 Citation

If you find StableProt useful in your research, please cite:

```bibtex
@article{behera2026stableprot,
  title={StableProt: Structure-Aware Deep Learning for Protein Thermostability ($T_m$) and Environmental Adaptation ($OGT$) Prediction},
  author={Behera, Bibhu Prasad and Dixit, Anshuman},
  journal={Working Manuscript},
  year={2026},
  publisher={Computational Biology and Bioinformatics Laboratory, iBRIC--Institute of Life Sciences},
  url={https://github.com/Bibhuprasadbehera/StableProt}
}
```

---

## 📬 Contact & Affiliation

**Computational Biology and Bioinformatics Laboratory**  
*iBRIC–Institute of Life Sciences (ILS), Bhubaneswar, Odisha, India*  
- **Bibhu Prasad Behera**: `bibhu.prasad@ils.res.in`  
- **Dr. Anshuman Dixit**: `anshuman@ils.res.in`