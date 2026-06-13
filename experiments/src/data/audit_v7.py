#!/usr/bin/env python3
"""Comprehensive audit of V7 pipeline: data, embeddings, models, evaluation."""
import torch
import os
import json
import numpy as np
from pathlib import Path

ROOT = Path("/home/bibhu/Documents/temstamsto")
# Fix path
ROOT = Path("/home/bibhu/Documents/temstampto")

print("=" * 70)
print("  COMPREHENSIVE V7 AUDIT")
print("=" * 70)

# ─────────────────────────────────────────────
# 1. CHECK TRAINING DATA FILE
# ─────────────────────────────────────────────
print("\n[1] TRAINING DATA FILE")
train_file = ROOT / "data/embeddings/prepared_data_v7_saprot1.3b_seqonly.pt"
if not train_file.exists():
    print("  CRITICAL: Training file does not exist!")
else:
    d = torch.load(train_file, map_location="cpu", weights_only=False)
    for k in d:
        keys = list(d[k].keys())
        print(f"  {k}: keys={keys}")
        emb = d[k].get("embeddings")
        if emb is not None:
            print(f"    embeddings shape: {emb.shape}, dtype: {emb.dtype}")
        tm = d[k].get("tm_consensus")
        if tm is not None:
            print(f"    tm_consensus: len={len(tm)}, range=[{tm.min():.1f}, {tm.max():.1f}]")
        ogt = d[k].get("ogt_consensus")
        if ogt is not None:
            print(f"    ogt_consensus: len={len(ogt)}, range=[{ogt.min():.1f}, {ogt.max():.1f}]")
        corr = d[k].get("ogt_bacdive_corrected")
        if corr is not None:
            print(f"    ogt_bacdive_corrected: {corr}")

# ─────────────────────────────────────────────
# 2. CHECK BACDIVE CORRECTION STATUS
# ─────────────────────────────────────────────
print("\n[2] BACDIVE CORRECTION STATUS")
bacdive_flag = d.get("train_ogt", {}).get("ogt_bacdive_corrected", None)
if bacdive_flag:
    print("  ✅ BacDive correction flag is set in training data")
else:
    print("  ❌ BacDive correction NOT applied to training data!")

# Compare with original v4
v4_file = ROOT / "data/embeddings/prepared_data_v4_saprot.pt"
if v4_file.exists():
    v4 = torch.load(v4_file, map_location="cpu", weights_only=False)
    orig_ogt = v4["train_ogt"]["ogt_consensus"]
    new_ogt = d["train_ogt"]["ogt_consensus"]
    diff = (orig_ogt - new_ogt).abs()
    changed = (diff > 0.01).sum().item()
    print(f"  OGT labels changed vs v4: {changed}/{len(orig_ogt)} ({100*changed/len(orig_ogt):.1f}%)")
    if changed > 0:
        print(f"  Mean change where changed: {diff[diff > 0.01].mean():.2f}°C")
    del v4

# ─────────────────────────────────────────────
# 3. CHECK EMBEDDING DIMENSION CONSISTENCY
# ─────────────────────────────────────────────
print("\n[3] EMBEDDING DIMENSION CONSISTENCY")
dims = {}
for split in ["train_tm", "val_tm", "test_tm", "train_ogt"]:
    emb = d[split].get("embeddings")
    if emb is not None:
        dims[split] = emb.shape[1]
        print(f"  {split}: dim={emb.shape[1]}, count={emb.shape[0]}")

unique_dims = set(dims.values())
if len(unique_dims) == 1:
    print(f"  ✅ All splits have consistent dim: {unique_dims.pop()}")
else:
    print(f"  ❌ DIMENSION MISMATCH: {dims}")

# ─────────────────────────────────────────────
# 4. CHECK TRAINED MODEL ARCHITECTURE
# ─────────────────────────────────────────────
print("\n[4] TRAINED MODEL CHECK")
import sys
sys.path.append(str(ROOT / "experiments/src/training/v7_shared"))
from train import MultiHeadSaProtV7

results_dir = ROOT / "experiments/src/training/v7_shared/results"
if results_dir.exists():
    for s in range(1, 6):
        model_path = results_dir / f"seed{s}/best_model.pt"
        if model_path.exists():
            state = torch.load(model_path, map_location="cpu", weights_only=False)
            input_dim = state["shared_layer1.weight"].shape[1]
            hidden1 = state["shared_layer1.weight"].shape[0]
            hidden2 = state["shared_layer2.weight"].shape[0]
            print(f"  seed{s}: input={input_dim}, h1={hidden1}, h2={hidden2}")
        else:
            print(f"  seed{s}: MISSING")

    config_path = results_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print(f"  Config: input_dim={config.get('input_dim')}, lr={config.get('lr')}")

    summary_path = results_dir / "ensemble_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        print(f"  Ensemble summary: {json.dumps(summary, indent=2)}")

# ─────────────────────────────────────────────
# 5. CHECK PROTHERMDB EVALUATION COVERAGE
# ─────────────────────────────────────────────
print("\n[5] PROTHERMDB EMBEDDING COVERAGE")
# Load ProThermDB sequences
protherm_path = ROOT / "data/test_data/protherm_validation_data.pt"
if protherm_path.exists():
    pt = torch.load(protherm_path, map_location="cpu", weights_only=False)
    protherm_seqs = pt.get("sequences", [])
    print(f"  ProThermDB sequences: {len(protherm_seqs)}")

    # Check how many match V7 training data
    v7_seqs = set()
    for split in ["train_tm", "val_tm", "test_tm"]:
        if "sequences" in d[split]:
            for seq in d[split]["sequences"]:
                v7_seqs.add(str(seq).upper())
    
    matched = sum(1 for seq in protherm_seqs if str(seq).upper() in v7_seqs)
    missing = len(protherm_seqs) - matched
    print(f"  Matched to V7 training seqs: {matched}/{len(protherm_seqs)}")
    print(f"  Missing (will fallback to V5): {missing} ({100*missing/len(protherm_seqs):.1f}%)")
    if missing > 0:
        print(f"  ❌ {missing} ProThermDB sequences have NO V7 embeddings!")
else:
    print("  WARNING: protherm_validation_data.pt not found")

# Check dedicated ProThermDB SaProt embeddings
protherm_saprot = ROOT / "data/embeddings/saprot_1.3b/protherm_embeddings.pt"
if protherm_saprot.exists():
    pt_emb = torch.load(protherm_saprot, map_location="cpu", weights_only=False)
    print(f"  ProThermDB SaProt embeddings file: shape={pt_emb.shape}")
else:
    print("  ProThermDB SaProt embeddings file: NOT FOUND")

# ─────────────────────────────────────────────
# 6. CHECK FIREPROT EVALUATION COVERAGE
# ─────────────────────────────────────────────
print("\n[6] FIREPROT EMBEDDING COVERAGE")
fireprot_path = ROOT / "data/test_data/fireprot_holdout_saprot.pt"
if fireprot_path.exists():
    fp = torch.load(fireprot_path, map_location="cpu", weights_only=False)
    print(f"  FireProt keys: {list(fp.keys())}")
    saprot_emb = fp.get("embeddings_saprot")
    if saprot_emb is not None:
        print(f"  embeddings_saprot: shape={saprot_emb.shape}, dim={saprot_emb.shape[1]}")
    else:
        print("  ❌ embeddings_saprot NOT found in fireprot holdout file!")

# ─────────────────────────────────────────────
# 7. CHECK EVAL SCRIPT LOGIC
# ─────────────────────────────────────────────
print("\n[7] EVALUATION SCRIPT LOGIC CHECK")
eval_protherm = ROOT / "experiments/src/eval/evaluate_all_models_protherm.py"
if eval_protherm.exists():
    with open(eval_protherm) as f:
        content = f.read()
    
    # Check if V7 uses correct task= kwarg
    if 'task="tm"' in content or "task='tm'" in content:
        print("  ✅ ProTherm eval uses task='tm' for V7")
    else:
        print("  ❌ ProTherm eval may use wrong kwarg (head= vs task=)")
    
    # Check fallback logic
    if "V5 Multi-Head" in content and "v7_full" in content:
        print("  ⚠️  V7 ProTherm eval falls back to V5 for missing sequences")

eval_fireprot = ROOT / "experiments/src/eval/evaluate_all_models_fireprot.py"
if eval_fireprot.exists():
    with open(eval_fireprot) as f:
        content = f.read()
    
    if 'task="tm"' in content or "task='tm'" in content:
        print("  ✅ FireProt eval uses task='tm' for V7")
    else:
        print("  ❌ FireProt eval may use wrong kwarg")

# ─────────────────────────────────────────────
# 8. CHECK ESMFold PROGRESS
# ─────────────────────────────────────────────
print("\n[8] ESMFOLD PDB STATUS")
structures_dir = ROOT / "data/structures"
if structures_dir.exists():
    pdb_files = list(structures_dir.glob("*.pdb"))
    print(f"  PDB files in data/structures/: {len(pdb_files)}")
else:
    print("  data/structures/ does not exist")

# Check Tm sequences that need structures
tm_seqs = d["train_tm"]["sequences"]
print(f"  Total Tm train sequences needing structures: {len(tm_seqs)}")
val_seqs = d["val_tm"]["sequences"]
test_seqs = d["test_tm"]["sequences"]
print(f"  Total Tm val sequences: {len(val_seqs)}")
print(f"  Total Tm test sequences: {len(test_seqs)}")
total_tm = len(tm_seqs) + len(val_seqs) + len(test_seqs)
print(f"  Total Tm needing ESMFold: {total_tm}")

# ─────────────────────────────────────────────
# 9. CHECK FLIP MELTOME DATA
# ─────────────────────────────────────────────
print("\n[9] FLIP MELTOME STATUS")
flip_dir = ROOT / "data" / "flip_meltome"
if flip_dir.exists():
    print(f"  FLIP directory exists: {list(flip_dir.iterdir())}")
else:
    print("  ❌ FLIP Meltome data NOT downloaded")

# ─────────────────────────────────────────────
# 10. SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("  AUDIT SUMMARY")
print("=" * 70)

del d
