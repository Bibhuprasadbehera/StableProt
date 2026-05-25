#!/bin/bash
set -e

PYTHON="/home/bibhu/miniconda3/envs/stableprot_v2/bin/python"
BASE_DIR="/home/bibhu/Documents/temstampto"

echo "=== Training ESM-2 Models (D1-D4) ==="

for mode in D1 D2 D3 D4; do
    echo -e "\n--- Training Mode $mode ---"
    $PYTHON "$BASE_DIR/experiments/src/training/v7_transfer/train_v3.py" --mode $mode
done

echo -e "\n=== Running ESM-2 Evaluation on FireProt Holdout ==="
$PYTHON "$BASE_DIR/experiments/src/training/v7_transfer/evaluate_v3.py"

echo -e "\n=== All ESM-2 Training and Evaluation Done! ==="
