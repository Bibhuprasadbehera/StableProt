# Use official PyTorch runtime with CUDA 12.1 support
# Works automatically on both NVIDIA GPUs and CPU-only systems
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    MODELS_DIR=/app/models \
    DEVICE=auto \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

# Install minimal system utilities for networking & health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (layer cached)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy model checkpoints and application code
COPY models /app/models
COPY app /app/app

# Expose port 8000 for web interface and REST API
EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI server with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
