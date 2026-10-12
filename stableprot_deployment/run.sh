#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "========================================================="
echo "       StableProt Web Suite & Prediction Server          "
echo "========================================================="

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo "Docker detected."
    if docker compose version &> /dev/null; then
        echo "Starting with Docker Compose..."
        docker compose up --build
        exit 0
    elif command -v docker-compose &> /dev/null; then
        echo "Starting with legacy docker-compose..."
        docker-compose up --build
        exit 0
    else
        echo "Building and running standalone Docker container..."
        docker build -t stableprot:latest .
        docker run --rm -it -p 8000:8000 --gpus all stableprot:latest
        exit 0
    fi
fi

# Direct Python fallback
echo "Docker not found or not in PATH. Checking Python environment..."
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    if command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        echo "Error: Neither Docker nor Python is installed."
        exit 1
    fi
fi

echo "Using: $($PYTHON_CMD --version)"
echo "Installing/verifying requirements..."
$PYTHON_CMD -m pip install -r requirements.txt

echo "Starting StableProt server on http://localhost:8000 ..."
export MODELS_DIR="$DIR/models"
export DEVICE="auto"
$PYTHON_CMD -m uvicorn app.main:app --host 0.0.0.0 --port 8000
