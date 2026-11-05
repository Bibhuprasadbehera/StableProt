<p align="center">
  <img src="assets/flame_with_text.png" alt="StableProt Logo" width="460">
</p>

<p align="center">
  <b>Structure-Aware Protein Thermostability (<i>T<sub>m</sub></i>) & Environmental Adaptation (<i>OGT</i>) Deep Learning with Calibrated Uncertainty</b>
</p>

<p align="center">
  <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat-square&logo=pytorch" alt="PyTorch"></a>
  <a href="https://github.com/westlake-repl/SaProt"><img src="https://img.shields.io/badge/SaProt-650M%20Transformer-0074B8.svg?style=flat-square" alt="SaProt"></a>
  <a href="https://fastapi.tiangolo.com"><img src="https://img.shields.io/badge/FastAPI-Web%20Application-009688.svg?style=flat-square&logo=fastapi" alt="FastAPI"></a>
  <a href="https://huggingface.co/Bibhuprasadbehera/StableProt"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Weights-orange.svg?style=flat-square" alt="Hugging Face"></a>
  <a href="#benchmark-performance-summary"><img src="https://img.shields.io/badge/Evaluation-Strict%20Homology%20Audit%20(<30%25)-EAAC08.svg?style=flat-square" alt="Audit"></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-MIT-0F172A.svg?style=flat-square" alt="License"></a>
</p>

---

## 🌟 Key Highlights

1. **Structure-Aware 3Di Tokenization**: Fuses primary amino acid sequences with Foldseek 3Di geometric structural states through a frozen SaProt 650M foundation backbone, explicitly capturing tertiary contacts, packing density, and backbone conformational strain without requiring experimental crystal structures.
2. **Decoupled Multi-Head Architecture**: Parameter-isolated neural pathways eliminate destructive gradient interference ($\cos\theta = -0.077 \to 0$) between organismal growth limits ($OGT$) and single-molecule melting temperatures ($T_m$), feeding an environmental prior link ($\hat{y}_{\text{OGT}} \to T_m$) during inference.
3. **Calibrated 95% Predictive Intervals**: Replaces uncalibrated point estimates with observation-level heteroscedastic uncertainty ($\mu \pm 1.96 c^* \sigma$) via a 5-model deep ensemble and post-hoc variance scaling, reducing Expected Calibration Error (ECE) to **1.92%**.
4. **Targeted Mesophilic Subsampling (14%)**: Dynamically counteracts extreme database imbalance by subsampling mesophilic data to 14% per epoch while preserving 100% of psychrophiles and thermophiles. This expands thermophile representation by 2.3-fold (16.8% &rarr; 38.0%) and mitigates mesophilic collapse on high-temperature biocatalysts ($T_m \ge 80^\circ\text{C}$).
5. **Prospective Industrial Enzyme Validation**: Validated across 82 non-redundant industrial enzymes (84 records), achieving a **75.3%** inclusion rate within calibrated 95% confidence intervals (mean half-width $\pm 6.1^\circ\text{C}$), demonstrating practical utility for de-risking wet-lab candidate screening.
6. **Interactive Web Suite & Loop Design Studio**: Production-ready FastAPI interface providing real-time dual-target predictions, calibrated thermometers, secondary structure profiling (Chou–Fasman scales), and in silico mutagenesis for surface loop stabilization.

---

## 📊 Benchmark Performance Summary

All evaluations enforce strict **bidirectional homology decontamination ($<30\%$ sequence identity)** between training and evaluation partitions to ensure zero data contamination.

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
| **Prospective Biocatalysts ($N=82$)** | 95% CI Inclusion Rate $\uparrow$ | — | — | — | — | **75.3%** (61/81 scored) |

*Note: Evaluated across 82 non-redundant industrial enzymes (84 records; 81 scored against reference experimental endpoints, 1 blind target CaKC16B) with sharp mean half-width of $\pm 6.1^\circ\text{C}$. For deterministic baselines lacking predictive intervals, CRPS equals their exact MAE without mathematical penalty.*

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

### 2. Pre-Trained Weights

Model weights for the 5-seed deep ensemble are available on Hugging Face:
- **Repository**: [https://huggingface.co/Bibhuprasadbehera/StableProt](https://huggingface.co/Bibhuprasadbehera/StableProt)

To download via Python:
```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Bibhuprasadbehera/StableProt",
    local_dir="huggingface_model_weights"
)
```

### 3. Launching the Interactive Web Suite

You can launch the web application directly using Uvicorn:

```bash
# Option A: From the repository root
python -m uvicorn inference.main:app --host 0.0.0.0 --port 8000

# Option B: Using the self-contained deployment package
cd stableprot_deployment
bash run_server.sh
```

Open **`http://localhost:8000`** in your browser to access:
- **Home**: Overview of StableProt capabilities in a clean, modern interface.
- **Predict**: Instant $T_m$ and $OGT$ predictions with calibrated confidence bounds and dynamic thermometers.
- **Design (Loop Design Studio)**: Automated Chou–Fasman secondary structure profiling, flexible loop detection, and real-time in silico point mutagenesis evaluation.
- **About**: Architectural foundation, parameter decoupling, uncertainty calibration, and citation info.

Alternatively, launch with Docker:
```bash
cd stableprot_deployment
docker compose up --build
```

---

## 💻 Programmatic Python API

```python
from inference.v9_predict import V9Predictor

# Initialize the 5-seed production ensemble predictor
predictor = V9Predictor(
    models_dir="experiments/src/training/v9_disjoint/results",
    device="cuda"  # or "cpu"
)

# Predict on a candidate protein sequence
sequence = "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQHKLRKLNPPDESGPGCMSCKCVLS"
result = predictor.predict_single(sequence)

print(f"Melting Temperature (Tm): {result['tm_pred']:.2f} ± {result['tm_conf']:.2f} °C")
print(f"Optimal Growth Temp (OGT): {result['ogt_pred']:.2f} ± {result['ogt_conf']:.2f} °C")
print(f"Calibrated 95% CI: [{result['ci_low']:.2f} °C, {result['ci_high']:.2f} °C]")
print(f"Thermal Tier: {result.get('thermal_tier', 'Mesophilic')}")
```

### Secondary Structure & Loop Profiling:

```python
from inference.v9_predict import predict_secondary_structure

# Identify flexible surface loops via Chou-Fasman propensity scales
ss_profile = predict_secondary_structure(sequence)
print("Secondary structure string:", ss_profile["ss_str"][:30], "...")
print(f"Detected {len(ss_profile['loops'])} flexible surface loops.")
```

---

## 📂 Repository Structure

```text
StableProt/
├── inference/                     # Production inference pipelines & FastAPI server
│   ├── main.py                    # Web server entrypoint
│   ├── v9_predict.py              # 5-seed ensemble predictor & Loop Design logic
│   ├── templates/                 # Modern interactive Jinja2 HTML UI
│   └── static/                    # Stylesheets, icons, and assets
├── stableprot_deployment/         # Self-contained deployment package (Docker / Uvicorn)
│   ├── Dockerfile                 # Container specification
│   ├── docker-compose.yml         # Container orchestration
│   ├── run_server.sh              # One-command startup script
│   └── app/                       # Server application & templates
├── experiments/                   # Training, calibration, and benchmarking experiments
│   ├── src/training/v9_disjoint/  # Disjoint dual-head training routines & checkpoints
│   └── src/eval/                  # Homology-decontaminated benchmark evaluation scripts
├── huggingface_model_weights/     # Pre-trained ensemble weights (seeds 1-5 + normalization)
├── manuscript/                    # Manuscript drafts, supplementary tables & audits
├── requirements.txt               # Python package dependencies
└── README.md                      # Repository overview & documentation
```

---

## 📖 Citation

If you find StableProt useful in your research or biocatalyst engineering workflows, please cite:

```bibtex
@article{behera2026stableprot,
  title={StableProt: Structure-Aware Deep Learning for Protein Thermostability ($T_m$) and Environmental Adaptation ($OGT$) Prediction with Calibrated Uncertainty},
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
*iBRIC–Institute of Life Sciences (ILS), Bhubaneswar, Odisha, 751023, India*

- **Bibhu Prasad Behera**: [`bibhu.prasad@ils.res.in`](mailto:bibhu.prasad@ils.res.in)
- **Dr. Anshuman Dixit**: [`anshuman@ils.res.in`](mailto:anshuman@ils.res.in)
- **GitHub**: [https://github.com/Bibhuprasadbehera/StableProt](https://github.com/Bibhuprasadbehera/StableProt)
- **Hugging Face**: [https://huggingface.co/Bibhuprasadbehera/StableProt](https://huggingface.co/Bibhuprasadbehera/StableProt)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).