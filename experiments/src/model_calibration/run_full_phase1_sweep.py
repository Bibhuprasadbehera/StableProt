#!/usr/bin/env python3
"""
Complete 157-Run Phase 1 Diagnostic Sweep Master Orchestrator (In-Memory Fast Edition)
(`experiments/src/model_calibration/run_full_phase1_sweep.py`)

Preloads all embedding datasets into RAM ONCE (`~14 seconds`), computes/caches string features,
and executes all remaining configurations with continuous GPU utilization and unbuffered logging.
"""

import os
import sys
import time
import torch
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT / "experiments" / "src" / "model_calibration"))
import sweep_runner

LOGS_DIR = PROJECT_ROOT / "experiments" / "src" / "model_calibration" / "logs"
CSV_PATH = LOGS_DIR / "summary_metrics.csv"

# Define all 157 planned configurations
SWEEP_CONFIGS = [
    # Group 1: Data (28 runs)
    *([('data', 'ogt_subsample_meso_rate', v, 1) for v in [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.0]]),
    *([('data', 'filter_tm_lt_ogt', v, 1) for v in ['remove', 'downweight_0.5', 'downweight_0.3', 'none']]),
    *([('data', 'seq_len_min', v, 1) for v in [50, 100, 150, 200]]),
    *([('data', 'iqr_filter_max', v, 1) for v in [1.5, 2.0, 2.5, 3.0]]),

    # Group 2: Loss (39 runs)
    *([('loss', 'huber_delta_ogt', v, 1) for v in [5, 8, 10, 12, 15, 18, 20, 25]]),
    *([('loss', 'focal_gamma', v, 1) for v in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]]),
    *([('loss', 'focal_beta', v, 1) for v in [0.1, 0.3, 0.5, 0.7, 1.0]]),
    *([('loss', 'weight_clamp_max', v, 1) for v in [5, 10, 15, 20, 22, 25, 30, 50]]),
    *([('loss', 'weight_power', v, 1) for v in [0.0, 0.5, 0.75, 1.0, 1.25, 1.5]]),
    *([('loss', 'loss_tm_type', v, 1) for v in ['nll_softplus', 'huber_5', 'huber_10', 'mse', 'mae', 'quantile']]),

    # Group 3: Architecture (22 runs)
    *([('arch', 'mlp_layers', v, 1) for v in [2, 3, 4, 5]]),
    *([('arch', 'hidden_size_1', v, 1) for v in [128, 256, 512, 768, 1024]]),
    *([('arch', 'dropout_1', v, 1) for v in [0.1, 0.2, 0.3, 0.4, 0.5]]),
    *([('arch', 'proj_dim', v, 1) for v in [32, 64, 128, 256]]),
    *([('arch', 'norm_type', v, 1) for v in ['layernorm', 'batchnorm']]),
    *([('arch', 'use_residuals', v, 1) for v in ['True', 'False']]),

    # Group 4: Optimization (27 runs)
    *([('optim', 'learning_rate', v, 1) for v in ['1e-5', '3e-5', '5e-5', '1e-4', '3e-4', '5e-4', '1e-3']]),
    *([('optim', 'batch_size', v, 1) for v in [16, 32, 64, 128, 256]]),
    *([('optim', 'weight_decay', v, 1) for v in ['1e-6', '5e-6', '1e-5', '5e-5', '1e-4', '5e-4']]),
    *([('optim', 'grad_clip_max_norm', v, 1) for v in [0.5, 1.0, 2.0, 5.0, 100.0]]),
    *([('optim', 'warmup_epochs', v, 1) for v in [0, 2, 5, 10]]),

    # Group 5: Calibration (8 runs)
    *([('calib', 'nll_softplus_eps', v, 1) for v in ['1e-5', '1e-4', '1e-3', '1e-2']]),
    *([('calib', 'ensemble_seeds', v, v) for v in [1, 3, 5, 7]]),

    # Group 6: Augmentation (27 runs)
    *([('aug', 'mixup_alpha', v, 1) for v in [0.0, 0.1, 0.2, 0.3, 0.5]]),
    *([('aug', 'augment_prob', v, 1) for v in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]]),
    *([('aug', 'augment_noise_std', v, 1) for v in [0.01, 0.02, 0.03, 0.05]]),
    *([('aug', 'target_jitter_std', v, 1) for v in [0.0, 0.3, 0.5, 0.8, 1.0, 2.0]]),
    *([('aug', 'tm_ogt_noise_std', v, 1) for v in [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]]),

    # Group 7: Features (8 runs)
    *([('feat', 'use_aa_ratios', v, 1) for v in ['True', 'False']]),
    *([('feat', 'use_ogt_prior', v, 1) for v in ['True', 'False']]),
    *([('feat', 'use_tmhmm', v, 1) for v in ['True', 'False']]),
    *([('feat', 'use_seq_len', v, 1) for v in ['True', 'False']]),

    # Group 8: Embedding (8 runs)
    *([('emb', 'backbone_type', v, 1) for v in ['saprot_650m_last', 'saprot_650m_layer30', 'saprot_650m_layer33', 'saprot_1.3b_last', 'saprot_1.3b_layer30']]),
    *([('emb', 'pooling_method', v, 1) for v in ['mean', 'cls', 'attention']]),
]

def load_completed_runs():
    if not CSV_PATH.exists():
        return set()
    try:
        df = pd.read_csv(CSV_PATH)
        completed = set()
        for _, row in df.iterrows():
            completed.add((str(row['group']), str(row['param']), str(row['value']), int(row['seed'])))
        return completed
    except Exception as e:
        print(f"Warning reading CSV: {e}", flush=True)
        return set()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Master Phase 1 Sweep Orchestrator (In-Memory Edition)")
    parser.add_argument("--dry_run", action="store_true", help="Print total count and pending runs without executing")
    args = parser.parse_args()

    print(f"\n=======================================================", flush=True)
    print(f"COMPLETE PHASE 1 DIAGNOSTIC SWEEP ORCHESTRATOR (FAST)", flush=True)
    print(f"Total Planned Configurations: {len(SWEEP_CONFIGS)}", flush=True)
    print(f"=======================================================\n", flush=True)

    completed = load_completed_runs()
    pending = []
    for g, p, v, s in SWEEP_CONFIGS:
        if (str(g), str(p), str(v), int(s)) not in completed:
            pending.append((g, p, v, s))

    print(f"Status: {len(completed)} completed, {len(pending)} pending out of {len(SWEEP_CONFIGS)} total runs.\n", flush=True)

    if args.dry_run:
        print("Dry run mode: showing first 15 pending runs:", flush=True)
        for g, p, v, s in pending[:15]:
            print(f"  [{g}] {p} = {v} (seed={s})", flush=True)
        return

    os.makedirs(LOGS_DIR, exist_ok=True)

    print("Preloading embedding datasets into RAM...", flush=True)
    t0 = time.time()
    preloaded_data = {}
    preloaded_data['saprot_650m'] = torch.load("data/embeddings/saprot_tm_struct_embeddings.pt", map_location='cpu', weights_only=False)
    ogt_split = torch.load("data/embeddings/prepared_data_v7_saprot1.3b_seqonly_ogt_split.pt", map_location='cpu', weights_only=False)
    preloaded_data['saprot_650m']['val_ogt'] = ogt_split['val_ogt']

    if os.path.exists("data/embeddings/prepared_data_v7_saprot1.3b_seqonly.pt"):
        preloaded_data['saprot_1.3b'] = torch.load("data/embeddings/prepared_data_v7_saprot1.3b_seqonly.pt", map_location='cpu', weights_only=False)
        preloaded_data['saprot_1.3b']['val_ogt'] = ogt_split['val_ogt']
    print(f"Preloading complete in {time.time() - t0:.1f}s. Beginning continuous GPU sweep loop...\n", flush=True)

    for i, (g, p, v, s) in enumerate(pending, 1):
        print(f"\n--- Launching Run {i}/{len(pending)} (Total progress: {len(completed)+i}/{len(SWEEP_CONFIGS)}) ---", flush=True)
        try:
            sweep_runner.run_single_sweep(g, p, v, seed=s, epochs=15, patience=5, debug=False, preloaded_data=preloaded_data)
        except Exception as e:
            print(f"Run failed for [{g}] {p}={v}: {e}", flush=True)
        except KeyboardInterrupt:
            print("\nSweep interrupted by user. Exiting cleanly. Progress saved to summary_metrics.csv.", flush=True)
            break

    print("\nPhase 1 Master Sweep Orchestrator loop completed.", flush=True)

if __name__ == "__main__":
    main()
