# StableProt Web Suite & Prediction Server

Self-contained, production-ready deployment package for the **StableProt** protein stability prediction and loop design platform.

---

## ⚡ Quick Start

### Option 1: Docker Compose (Recommended)

To run the container with automatic GPU detection (or CPU fallback) and persistent model caching:

```bash
docker compose up --build
```

Once running, access the web interface at:
**[http://localhost:8000](http://localhost:8000)**

To run in detached (background) mode:
```bash
docker compose up -d
```

To stop:
```bash
docker compose down
```

---

### Option 2: Standalone Docker Run

1. **Build the image**:
   ```bash
   docker build -t stableprot:latest .
   ```

2. **Run on GPU**:
   ```bash
   docker run --rm -p 8000:8000 --gpus all stableprot:latest
   ```

3. **Run on CPU-only machine**:
   ```bash
   docker run --rm -p 8000:8000 stableprot:latest
   ```

> **Tip (Cache SaProt to Avoid Re-Downloading):**
> Mount your host's HuggingFace cache directory:
> ```bash
> docker run --rm -p 8000:8000 --gpus all \
>   -v ~/.cache/huggingface:/root/.cache/huggingface \
>   stableprot:latest
> ```

---

### Option 3: Direct Python (No Docker)

If you prefer running directly on a Linux/macOS/Windows host with Python 3.10+:

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Or simply run `./run.sh`.

---

## 💻 Hardware & Resource Requirements

StableProt couples a **650M-parameter structure-aware language model (SaProt)** with a **5-seed dual-head neural ensemble (Tm + OGT)**.

### Resource Matrix

| Protein Length | GPU VRAM (Inference) | GPU Latency | CPU RAM | CPU Latency |
| :--- | :--- | :--- | :--- | :--- |
| **Shorter Sequences** (50 – 300 aa) | **~2.5 – 3.5 GB** | **~0.15 – 0.35 s** | ~6 – 8 GB | ~1.0 – 2.5 s |
| **Medium Sequences** (300 – 600 aa) | **~3.5 – 5.0 GB** | **~0.35 – 0.70 s** | ~8 – 10 GB | ~2.5 – 6.0 s |
| **Longer Sequences** (600 – 1,024 aa) | **~5.0 – 7.5 GB** | **~0.70 – 1.40 s** | ~10 – 14 GB | ~6.0 – 12.0 s |

### Disk Storage

- **Base Docker Image + PyTorch Runtime**: ~4.5 GB
- **Base Foundation Model (`SaProt_650M_AF2`)**: ~2.6 GB (downloaded once from HuggingFace)
- **StableProt Ensemble Checkpoints (`models/`)**: ~75 MB (pre-packaged)
- **Recommended Free Disk Space**: **~10 – 15 GB**

---

## 📦 How to Share / Transfer This Folder

This folder (`stableprot_deployment`) is 100% self-contained. You can zip or tar it and send it to any collaborator:

```bash
# Create tarball archive
tar -czvf stableprot_deployment.tar.gz stableprot_deployment/

# Or create zip archive
zip -r stableprot_deployment.zip stableprot_deployment/
```

The recipient only needs to extract the archive and run:
```bash
cd stableprot_deployment
docker compose up --build
```
Everything required (trained heads, normalization statistics, web interface, and API routes) will load automatically.

---

## 📡 REST API Endpoints

### 1. Health Check
- **Endpoint**: `GET /health`
- **Response**:
  ```json
  {
    "status": "ok",
    "model_loaded": true,
    "device": "cuda",
    "cuda_available": true
  }
  ```

### 2. Stability Prediction ($T_m$ & OGT)
- **Endpoint**: `POST /api/predict`
- **Payload**:
  ```json
  {
    "sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
  }
  ```
- **Response**:
  ```json
  {
    "status": "OK",
    "sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    "length": 65,
    "tm_pred": 77.85,
    "tm_conf": 13.00,
    "ci_low": 64.85,
    "ci_high": 90.85,
    "thermal_tier": "Thermophilic (60–80°C)",
    "ogt_pred": 69.99,
    "ogt_conf": 15.31,
    "aa_composition": {
      "hydrophobic": 35.4,
      "polar": 16.9,
      "charged": 27.7,
      "aromatic": 6.2
    }
  }
  ```

### 3. Secondary Structure & Loop Identification
- **Endpoint**: `POST /api/identify_loops`
- **Payload**: `{"sequence": "..."}`
- Returns per-residue secondary structure string (`H`=Helix, `E`=Sheet, `L`=Loop) and segmented loop regions with flexibility scores.

### 4. In Silico Mutant Evaluation
- **Endpoint**: `POST /api/evaluate_design`
- **Payload**:
  ```json
  {
    "wt_sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    "mutant_sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGP"
  }
  ```
- Returns $\Delta T_m$, classification (`Stabilizing` / `Destabilizing`), mutation labels (e.g. `G65P`), and whether mutations reside in flexible loop regions.

### 5. Automated Verification
Run the included verification script against a running server:
```bash
python3 test_api.py
```
