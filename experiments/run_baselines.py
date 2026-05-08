import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score, f1_score
import sys

def evaluate_temstapro_v0():
    print("--- Evaluating TemStaPro V0 ---")
    # Here we would load V0 model and run inference on ProThermDB sequences
    print("TemStaPro V0 is a binary classifier. We evaluate it at 40, 45, 50, 55, 60, 65°C thresholds.")
    print("Need to implement wrapper for v0_original inference.")
    
def evaluate_tembert_tm():
    print("\n--- Evaluating TemBERTureTm ---")
    repo_path = "../TemBERTure_repo"
    if not os.path.exists(repo_path):
        print(f"Error: {repo_path} not found. Clone TemBERTure first.")
        return
        
    print("Run inside the TemBERTure repo:")
    print("cd ../TemBERTure_repo && python -c 'from temBERTure import TemBERTure; print(\"Running TemBERTure...\")'")
    print("Need to adapt their inference script to take ProThermDB fasta and output CSV.")

if __name__ == "__main__":
    evaluate_temstapro_v0()
    evaluate_tembert_tm()
