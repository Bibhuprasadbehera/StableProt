#!/usr/bin/env python3
"""
Curate dedicated Extremophile Datasets for Optimal Growth Temperature (OGT) and Melting Temperature (Tm).

Extracts, validates, classifies thermal regimes, and exports:
1. CSV metadata tables (OGT benchmark, Tm benchmark, paired OGT-Tm, master catalog)
2. FASTA sequence files for pLM embeddings and inference
3. PyTorch .pt tensor bundles for direct benchmarking and fine-tuning with StableProt

Thermal Regimes:
- Psychrophilic: OGT < 20°C, Tm < 45°C
- Mesophilic Control: 20°C <= OGT <= 50°C, 45°C <= Tm < 65°C
- Moderate Thermophilic: 50°C < OGT <= 65°C, 65°C <= Tm < 75°C
- Extreme Thermophilic: 65°C < OGT <= 80°C, 75°C <= Tm < 85°C
- Hyperthermophilic: OGT > 80°C, Tm >= 85°C
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Thermal regime thresholds
REGIMES_OGT = {
    'psychrophile': lambda t: t < 20.0,
    'mesophile_control': lambda t: (t >= 20.0) & (t <= 50.0),
    'thermophile_moderate': lambda t: (t > 50.0) & (t <= 65.0),
    'thermophile_extreme': lambda t: (t > 65.0) & (t <= 80.0),
    'hyperthermophile': lambda t: t > 80.0
}

REGIMES_TM = {
    'psychro_labile': lambda t: t < 45.0,
    'mesophile_control': lambda t: (t >= 45.0) & (t < 65.0),
    'thermophile_moderate': lambda t: (t >= 65.0) & (t < 75.0),
    'thermophile_extreme': lambda t: (t >= 75.0) & (t < 85.0),
    'hyperthermophile': lambda t: t >= 85.0
}

def classify_regime_ogt(temp):
    if temp < 20.0:
        return 'psychrophile'
    elif temp <= 50.0:
        return 'mesophile_control'
    elif temp <= 65.0:
        return 'thermophile_moderate'
    elif temp <= 80.0:
        return 'thermophile_extreme'
    else:
        return 'hyperthermophile'

def classify_regime_tm(temp):
    if temp < 45.0:
        return 'psychro_labile'
    elif temp < 65.0:
        return 'mesophile_control'
    elif temp < 75.0:
        return 'thermophile_moderate'
    elif temp < 85.0:
        return 'thermophile_extreme'
    else:
        return 'hyperthermophile'

def sanitize_sequence(seq):
    return "".join([c for c in str(seq).upper() if c.isupper() and c.isalpha()])

def curate_ogt_benchmarks(output_dir):
    """Extract OGT extremophiles from BRENDA OOD, BacDive, and Master dataset."""
    print("\n[1/4] Curating OGT Extremophile Benchmarks...")
    records = []
    
    # 1. BRENDA OOD Benchmark
    brenda_path = PROJECT_ROOT / "new_data/brenda_ood_benchmark.csv"
    if brenda_path.exists():
        df_brenda = pd.read_csv(brenda_path)
        for idx, row in df_brenda.iterrows():
            seq = sanitize_sequence(row.get('sequence', ''))
            if len(seq) < 30:
                continue
            ogt = float(row.get('ogt', row.get('OGT', 0.0)))
            uid = str(row.get('uniprot_id', f"BRENDA_{idx}"))
            regime = classify_regime_ogt(ogt)
            records.append({
                'id': uid,
                'uniprot_id': uid,
                'sequence': seq,
                'target_temperature': ogt,
                'temperature_type': 'OGT',
                'thermal_regime': regime,
                'is_extremophile': regime != 'mesophile_control',
                'source': 'BRENDA_OOD',
                'dataset_partition': 'test_ood',
                'organism': row.get('organism', ''),
                'domain': row.get('domain', '')
            })
    print(f"  Processed {len(records)} records from BRENDA OOD.")
    
    # 2. Master OGT embeddings (validation and holdout samples)
    master_path = PROJECT_ROOT / "data/embeddings/saprot_tm_struct_embeddings.pt"
    bacdive_csv_path = PROJECT_ROOT / "data/ogt_labels_bacdive_corrected.csv"
    
    tax_org_map = {}
    if bacdive_csv_path.exists():
        try:
            df_bac = pd.read_csv(bacdive_csv_path)
            for _, r in df_bac.iterrows():
                tid = str(r['tax_id'])
                tax_org_map[tid] = (str(r.get('organism_name', '')), float(r.get('final_OGT', 0.0)))
        except Exception as e:
            print(f"  Warning: could not load BacDive CSV: {e}")

    # Load master OGT dataset
    if master_path.exists():
        print("  Loading Master OGT dataset embeddings...")
        m = torch.load(master_path, map_location='cpu', weights_only=False)
        train_ogt = m.get('train_ogt', {})
        seqs = train_ogt.get('sequences', [])
        ogts = train_ogt.get('ogt_consensus', [])
        ids = train_ogt.get('ids', [f"OGT_{i}" for i in range(len(seqs))])
        tax_ids = train_ogt.get('tax_id', [None] * len(seqs))
        
        # Subsample high-confidence extremophiles for benchmark / balanced training
        psychro_cnt = 0
        extreme_cnt = 0
        hyper_cnt = 0
        
        for i in range(len(seqs)):
            ogt_val = float(ogts[i]) if i < len(ogts) and ogts[i] is not None else None
            if ogt_val is None:
                continue
            seq = sanitize_sequence(seqs[i])
            if len(seq) < 30:
                continue
            regime = classify_regime_ogt(ogt_val)
            
            # Select all psychrophiles and hyperthermophiles, plus balanced thermophiles
            is_candidate = False
            if regime == 'psychrophile':
                is_candidate = True
                psychro_cnt += 1
            elif regime == 'hyperthermophile':
                is_candidate = True
                hyper_cnt += 1
            elif regime == 'thermophile_extreme' and extreme_cnt < 10000:
                is_candidate = True
                extreme_cnt += 1
                
            if is_candidate:
                tid = str(tax_ids[i]) if tax_ids[i] is not None else ''
                org_name = tax_org_map.get(tid, ('', ogt_val))[0]
                records.append({
                    'id': str(ids[i]) if i < len(ids) else f"BACDIVE_{i}",
                    'uniprot_id': str(ids[i]) if i < len(ids) else '',
                    'sequence': seq,
                    'target_temperature': ogt_val,
                    'temperature_type': 'OGT',
                    'thermal_regime': regime,
                    'is_extremophile': True,
                    'source': 'BacDive_Master',
                    'dataset_partition': 'reference_pool',
                    'organism': org_name,
                    'domain': 'Archaea' if ogt_val > 75.0 else 'Bacteria'
                })
        print(f"  Extracted {psychro_cnt} psychrophiles, {extreme_cnt} extreme thermophiles, {hyper_cnt} hyperthermophiles from Master OGT.")
        
    df_ogt = pd.DataFrame(records)
    df_ogt_bench = df_ogt[df_ogt['is_extremophile']].drop_duplicates(subset=['sequence'])
    print(f"  Total Unique OGT Extremophile Dataset size: {len(df_ogt_bench)} proteins")
    return df_ogt_bench

def curate_tm_benchmarks(output_dir):
    """Extract Tm extremophiles from ProThermDB, FireProtDB, and Master Tm dataset."""
    print("\n[2/4] Curating Tm Extremophile Benchmarks...")
    records = []
    
    # 1. ProThermDB validation set
    protherm_path = PROJECT_ROOT / "new_data/protherm_evaluation_results.pt"
    protherm_csv = PROJECT_ROOT / "new_data/prothermdb_validation.csv"
    protherm_fasta = PROJECT_ROOT / "new_data/prothermdb_validation.fasta"
    
    fasta_seqs = {}
    if protherm_fasta.exists():
        for rec in SeqIO.parse(str(protherm_fasta), "fasta"):
            clean_id = rec.id.split("|")[0]
            fasta_seqs[clean_id] = sanitize_sequence(rec.seq)
            fasta_seqs[rec.id] = sanitize_sequence(rec.seq)
            
    if protherm_path.exists():
        pt_data = torch.load(protherm_path, map_location='cpu', weights_only=False)
        y_true = np.array(pt_data['y_true'])
        
        # Load metadata from CSV if available
        uid_list = []
        if protherm_csv.exists():
            df_pt = pd.read_csv(protherm_csv)
            for _, r in df_pt.iterrows():
                if not pd.isna(r.get('Tm')):
                    uid_list.append((str(r.get('UniProt_ID', '')), r.get('Organism', '')))
        
        for idx, tm_val in enumerate(y_true):
            uid = uid_list[idx][0] if idx < len(uid_list) else f"PROTHERM_{idx}"
            org = uid_list[idx][1] if idx < len(uid_list) else ''
            seq = fasta_seqs.get(uid, '')
            regime = classify_regime_tm(float(tm_val))
            records.append({
                'id': f"ProThermDB_{idx}_{uid}",
                'uniprot_id': uid,
                'sequence': seq,
                'target_temperature': float(tm_val),
                'temperature_type': 'Tm',
                'thermal_regime': regime,
                'is_extremophile': regime != 'mesophile_control',
                'source': 'ProThermDB',
                'dataset_partition': 'validation_holdout',
                'organism': org,
                'domain': ''
            })
        print(f"  Processed {len(y_true)} records from ProThermDB (Extremophiles: {sum(r['is_extremophile'] for r in records)}).")

    # 2. FireProtDB zero-shot holdout
    fireprot_path = PROJECT_ROOT / "data/test_data/fireprot_holdout_saprot.pt"
    if fireprot_path.exists():
        fp_data = torch.load(fireprot_path, map_location='cpu', weights_only=False)
        fp_seqs = fp_data.get('sequences', [])
        fp_tms = fp_data.get('temperatures', fp_data.get('tm_consensus', fp_data.get('labels', [])))
        fp_ogts = fp_data.get('ogt', [None] * len(fp_seqs))
        
        fp_count = 0
        n_items = min(len(fp_seqs), len(fp_tms))
        for idx in range(n_items):
            tm_val = float(fp_tms[idx])
            seq = sanitize_sequence(fp_seqs[idx])
            ogt_val = float(fp_ogts[idx]) if idx < len(fp_ogts) and fp_ogts[idx] is not None else None
            regime = classify_regime_tm(tm_val)
            records.append({
                'id': f"FireProtDB_{idx}",
                'uniprot_id': f"FP_{idx}",
                'sequence': seq,
                'target_temperature': tm_val,
                'host_ogt': ogt_val,
                'temperature_type': 'Tm',
                'thermal_regime': regime,
                'is_extremophile': regime != 'mesophile_control',
                'source': 'FireProtDB',
                'dataset_partition': 'test_zero_shot',
                'organism': '',
                'domain': ''
            })
            fp_count += 1
        print(f"  Processed {fp_count} records from FireProtDB zero-shot holdout.")
        
    # 3. Master Tm dataset (train_tm, val_tm, test_tm)
    master_path = PROJECT_ROOT / "data/embeddings/saprot_tm_struct_embeddings.pt"
    if master_path.exists():
        m = torch.load(master_path, map_location='cpu', weights_only=False)
        for split_name in ['train_tm', 'val_tm', 'test_tm']:
            split_dict = m.get(split_name, {})
            s_seqs = split_dict.get('sequences', [])
            s_tms = split_dict.get('tm_consensus', [])
            s_ogts = split_dict.get('ogt', [None] * len(s_seqs))
            s_ids = split_dict.get('ids', [f"{split_name}_{i}" for i in range(len(s_seqs))])
            s_src = split_dict.get('source', ['Master_Tm'] * len(s_seqs))
            
            n_items = min(len(s_seqs), len(s_tms))
            for i in range(n_items):
                tm_val = float(s_tms[i]) if s_tms[i] is not None else None
                if tm_val is None:
                    continue
                seq = sanitize_sequence(s_seqs[i])
                if len(seq) < 30:
                    continue
                regime = classify_regime_tm(tm_val)
                ogt_val = float(s_ogts[i]) if i < len(s_ogts) and s_ogts[i] is not None else None
                
                records.append({
                    'id': str(s_ids[i]) if i < len(s_ids) else f"{split_name}_{i}",
                    'uniprot_id': str(s_ids[i]) if i < len(s_ids) else f"{split_name}_{i}",
                    'sequence': seq,
                    'target_temperature': tm_val,
                    'host_ogt': ogt_val,
                    'temperature_type': 'Tm',
                    'thermal_regime': regime,
                    'is_extremophile': regime != 'mesophile_control',
                    'source': str(s_src[i]) if i < len(s_src) else split_name,
                    'dataset_partition': split_name,
                    'organism': '',
                    'domain': ''
                })
        print("  Processed Master Tm dataset splits.")
        
    df_tm = pd.DataFrame(records)
    # Deduplicate by sequence and source
    df_tm_clean = df_tm.drop_duplicates(subset=['sequence', 'temperature_type', 'target_temperature'])
    df_tm_bench = df_tm_clean[df_tm_clean['is_extremophile']]
    print(f"  Total Unique Tm Extremophile Dataset size: {len(df_tm_bench)} proteins (All Tm records: {len(df_tm_clean)})")
    return df_tm_bench, df_tm_clean

def curate_paired_ogt_tm(df_tm_all):
    """Extract proteins with paired OGT host and experimental Tm unfolding points."""
    print("\n[3/4] Curating Paired OGT-Tm Extremophiles...")
    paired = df_tm_all[df_tm_all['host_ogt'].notna() & (df_tm_all['host_ogt'] > 0)].copy()
    paired['ogt_regime'] = paired['host_ogt'].apply(classify_regime_ogt)
    paired['tm_regime'] = paired['target_temperature'].apply(classify_regime_tm)
    paired['is_extremophile_pair'] = (paired['ogt_regime'] != 'mesophile_control') | (paired['tm_regime'] != 'mesophile_control')
    
    extremophile_paired = paired[paired['is_extremophile_pair']].copy()
    print(f"  Total paired OGT-Tm proteins: {len(paired)} (Extremophile pairs: {len(extremophile_paired)})")
    return extremophile_paired

def build_pytorch_bundle(df_ogt, df_tm, df_paired, output_dir):
    """Build fast, self-contained PyTorch tensor bundles for evaluation and training."""
    print("\n[4/4] Building PyTorch Tensor Bundles (.pt)...")
    
    # 1. Benchmark Evaluation Bundle
    # Load cached baseline predictions and embeddings for FireProt, ProTherm, BRENDA
    brenda_preds = {}
    brenda_embs = None
    b_pred_path = PROJECT_ROOT / "data/embeddings/brenda_ood_baseline_preds.pt"
    b_emb_path = PROJECT_ROOT / "data/embeddings/brenda_ood_saprot_embeddings.pt"
    if b_pred_path.exists():
        brenda_preds = torch.load(b_pred_path, map_location='cpu', weights_only=False)
    if b_emb_path.exists():
        brenda_embs = torch.load(b_emb_path, map_location='cpu', weights_only=False)
        
    protherm_preds = {}
    protherm_true = []
    pt_eval_path = PROJECT_ROOT / "new_data/protherm_evaluation_results.pt"
    if pt_eval_path.exists():
        pt_eval = torch.load(pt_eval_path, map_location='cpu', weights_only=False)
        protherm_preds = pt_eval.get('predictions', {})
        protherm_true = pt_eval.get('y_true', [])
        
    fireprot_preds = {}
    fireprot_true = []
    fp_eval_path = PROJECT_ROOT / "new_data/fireprot_evaluation_results.pt"
    if fp_eval_path.exists():
        fp_eval = torch.load(fp_eval_path, map_location='cpu', weights_only=False)
        fireprot_preds = fp_eval.get('predictions', {})
        fireprot_true = fp_eval.get('y_true', [])

    benchmark_bundle = {
        'metadata': {
            'description': 'StableProt Extremophile Benchmark Suite for OGT and Tm',
            'creation_time': '2026-09-02',
            'regimes': list(REGIMES_OGT.keys())
        },
        'ogt_benchmark': {
            'records': df_ogt.to_dict(orient='records'),
            'brenda_ood_embeddings': brenda_embs,
            'brenda_ood_baseline_predictions': brenda_preds
        },
        'tm_benchmark': {
            'records': df_tm.to_dict(orient='records'),
            'prothermdb_predictions': protherm_preds,
            'fireprotdb_predictions': fireprot_preds
        },
        'paired_benchmark': {
            'records': df_paired.to_dict(orient='records')
        }
    }
    
    pt_dir = output_dir / "pt"
    pt_dir.mkdir(parents=True, exist_ok=True)
    bench_pt_path = pt_dir / "extremophile_benchmark.pt"
    torch.save(benchmark_bundle, bench_pt_path)
    print(f"  Saved PyTorch benchmark bundle to {bench_pt_path} ({bench_pt_path.stat().st_size / 1e6:.2f} MB)")
    
    # 2. Curated Train/Val Split for specialized extremophile fine-tuning
    master_path = PROJECT_ROOT / "data/embeddings/saprot_tm_struct_embeddings.pt"
    if master_path.exists():
        m = torch.load(master_path, map_location='cpu', weights_only=False)
        train_ogt = m.get('train_ogt', {})
        train_tm = m.get('train_tm', {})
        
        # Filter extremophile indices from train_ogt and train_tm
        ogt_consensus = np.array(train_ogt.get('ogt_consensus', []))
        tm_consensus = np.array(train_tm.get('tm_consensus', []))
        
        ext_ogt_mask = (ogt_consensus < 20.0) | (ogt_consensus > 50.0)
        ext_tm_mask = (tm_consensus < 45.0) | (tm_consensus >= 65.0)
        
        ext_ogt_indices = np.where(ext_ogt_mask)[0]
        ext_tm_indices = np.where(ext_tm_mask)[0]
        
        print(f"  Extracted {len(ext_ogt_indices)} OGT and {len(ext_tm_indices)} Tm extremophile training tensors.")
        
        # Extract subset tensors
        train_val_bundle = {
            'metadata': {'description': 'Extremophile Training & Fine-Tuning Tensors'},
            'train_ogt_extremophile': {
                'embeddings': train_ogt['embeddings'][ext_ogt_indices],
                'sequences': [train_ogt['sequences'][i] for i in ext_ogt_indices],
                'ogt_consensus': ogt_consensus[ext_ogt_indices].tolist(),
                'ids': [train_ogt['ids'][i] for i in ext_ogt_indices] if 'ids' in train_ogt else []
            },
            'train_tm_extremophile': {
                'embeddings': train_tm['embeddings'][ext_tm_indices],
                'sequences': [train_tm['sequences'][i] for i in ext_tm_indices],
                'tm_consensus': tm_consensus[ext_tm_indices].tolist(),
                'ogt': [train_tm['ogt'][i] for i in ext_tm_indices] if 'ogt' in train_tm else [],
                'tmhmm_tm_binary': [train_tm['tmhmm_tm_binary'][i] for i in ext_tm_indices] if 'tmhmm_tm_binary' in train_tm else [],
                'ids': [train_tm['ids'][i] for i in ext_tm_indices] if 'ids' in train_tm else []
            }
        }
        train_pt_path = pt_dir / "extremophile_train_val.pt"
        torch.save(train_val_bundle, train_pt_path)
        print(f"  Saved PyTorch training bundle to {train_pt_path} ({train_pt_path.stat().st_size / 1e6:.2f} MB)")

def export_fasta_files(df_ogt, df_tm, output_dir):
    """Export FASTA files categorized by thermal regimes."""
    fasta_dir = output_dir / "fasta"
    fasta_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. OGT FASTA
    ogt_records = []
    for _, r in df_ogt.iterrows():
        seq = str(r['sequence']).strip()
        if len(seq) >= 30:
            rec_id = f"{r['id']}|OGT={r['target_temperature']:.1f}|Regime={r['thermal_regime']}"
            ogt_records.append(SeqRecord(Seq(seq), id=rec_id, description=""))
    SeqIO.write(ogt_records, str(fasta_dir / "extremophile_ogt.fasta"), "fasta")
    
    # 2. Tm FASTA
    tm_records = []
    for _, r in df_tm.iterrows():
        seq = str(r['sequence']).strip()
        if len(seq) >= 30:
            rec_id = f"{r['id']}|Tm={r['target_temperature']:.1f}|Regime={r['thermal_regime']}"
            tm_records.append(SeqRecord(Seq(seq), id=rec_id, description=""))
    SeqIO.write(tm_records, str(fasta_dir / "extremophile_tm.fasta"), "fasta")
    
    # 3. Hyperthermophiles (>80°C)
    hyper_records = [rec for rec in ogt_records + tm_records if "Regime=hyperthermophile" in rec.id]
    SeqIO.write(hyper_records, str(fasta_dir / "extremophile_hyperthermo.fasta"), "fasta")
    
    # 4. Psychrophiles (<20°C / <45°C)
    psychro_records = [rec for rec in ogt_records if "Regime=psychrophile" in rec.id] + [rec for rec in tm_records if "Regime=psychro_labile" in rec.id]
    SeqIO.write(psychro_records, str(fasta_dir / "extremophile_psychro.fasta"), "fasta")
    
    print(f"  Exported FASTA files: {len(ogt_records)} OGT, {len(tm_records)} Tm, {len(hyper_records)} Hyperthermo, {len(psychro_records)} Psychro.")

def generate_readme(df_ogt, df_tm, df_paired, output_dir):
    """Write comprehensive README documentation."""
    readme_path = output_dir / "README.md"
    
    ogt_regimes_summary = df_ogt['thermal_regime'].value_counts().to_dict()
    tm_regimes_summary = df_tm['thermal_regime'].value_counts().to_dict()
    
    content = f"""# StableProt Extremophile Dataset Suite (OGT & Tm)

A curated, multi-regime benchmark and training suite specifically designed to evaluate and enhance protein language models on **extreme thermal regimes** (psychrophiles, moderate thermophiles, extreme thermophiles, and hyperthermophiles).

---

## 1. Thermal Regime Breakdown

### Optimal Growth Temperature (OGT) Extremophiles
Total Records: **{len(df_ogt):,}** unique proteins

| Thermal Regime | Range (°C) | Protein Count | Description |
|:---|:---:|:---:|:---|
| **Psychrophilic** | < 20.0 | {ogt_regimes_summary.get('psychrophile', 0):,} | Cold-adapted organisms (deep sea, polar) |
| **Moderate Thermophilic** | 50.0 – 65.0 | {ogt_regimes_summary.get('thermophile_moderate', 0):,} | Industrial thermotolerant species |
| **Extreme Thermophilic** | 65.0 – 80.0 | {ogt_regimes_summary.get('thermophile_extreme', 0):,} | Hot springs, hydrothermal vents (*Thermus*) |
| **Hyperthermophilic** | > 80.0 | {ogt_regimes_summary.get('hyperthermophile', 0):,} | Extreme hyperthermophilic Archaea (*Pyrococcus*, *Sulfolobus*) |

### Melting Temperature (Tm) Extremophiles
Total Records: **{len(df_tm):,}** unique proteins

| Thermal Regime | Range (°C) | Protein Count | Description |
|:---|:---:|:---:|:---|
| **Psychro-labile / Low Tm** | < 45.0 | {tm_regimes_summary.get('psychro_labile', 0):,} | Highly flexible / heat-sensitive unfolding |
| **Moderate Thermophilic** | 65.0 – 75.0 | {tm_regimes_summary.get('thermophile_moderate', 0):,} | Thermostable biocatalysts |
| **Extreme Thermophilic** | 75.0 – 85.0 | {tm_regimes_summary.get('thermophile_extreme', 0):,} | Highly stable compact folds |
| **Hyperthermophilic** | ≥ 85.0 (up to 105°C) | {tm_regimes_summary.get('hyperthermophile', 0):,} | Extreme thermal unfolding resistance |

### Paired OGT–Tm Cohort
Total Co-Annotated Proteins: **{len(df_paired):,}** records with verified host environmental OGT and experimental unfolding Tm.

---

## 2. Directory Structure

```
data/extremophile_dataset/
├── README.md                                 # This specification
├── csv/
│   ├── extremophile_master_catalog.csv       # Unified catalog across all sources
│   ├── extremophile_ogt_benchmark.csv        # Dedicated OGT extremophile dataset
│   ├── extremophile_tm_benchmark.csv         # Dedicated Tm extremophile dataset
│   └── extremophile_paired_ogt_tm.csv        # Co-annotated OGT and Tm proteins
├── fasta/
│   ├── extremophile_ogt.fasta                # All OGT extremophile sequences
│   ├── extremophile_tm.fasta                 # All Tm extremophile sequences
│   ├── extremophile_hyperthermo.fasta        # Hyperthermophiles (>80°C / >=85°C)
│   └── extremophile_psychro.fasta            # Psychrophiles (<20°C / <45°C)
└── pt/
    ├── extremophile_benchmark.pt             # PyTorch bundle with cached SaProt embeddings & baseline predictions
    └── extremophile_train_val.pt             # PyTorch tensors for model fine-tuning
```

---

## 3. Usage Examples

### Loading with Pandas (Python)
```python
import pandas as pd

# Load OGT extremophiles
df_ogt = pd.read_csv("data/extremophile_dataset/csv/extremophile_ogt_benchmark.csv")
print("Hyperthermophiles count:", len(df_ogt[df_ogt['thermal_regime'] == 'hyperthermophile']))

# Load Tm extremophiles
df_tm = pd.read_csv("data/extremophile_dataset/csv/extremophile_tm_benchmark.csv")
print("Extreme thermostable Tm count:", len(df_tm[df_tm['target_temperature'] >= 75.0]))
```

### Loading PyTorch Benchmark Bundle
```python
import torch

data = torch.load("data/extremophile_dataset/pt/extremophile_benchmark.pt", map_location="cpu", weights_only=False)
ogt_bench = data['ogt_benchmark']
tm_bench = data['tm_benchmark']

print(f"Loaded {{len(ogt_bench['records'])}} OGT and {{len(tm_bench['records'])}} Tm benchmark records.")
```

---

## 4. Benchmark Evaluation
To evaluate StableProt vs literature baselines (TemBERTure, DeepSTABp, ESMStabP, ThermoFormer, PRIME) on this extremophile dataset, run:

```bash
/home/bibhu/miniconda3/envs/stableprot_v2/bin/python experiments/src/eval/evaluate_extremophile_benchmark.py
```
"""
    with open(readme_path, "w") as f:
        f.write(content)
    print(f"  Wrote documentation to {readme_path}")

def main():
    parser = argparse.ArgumentParser(description="Curate Extremophile Datasets for OGT and Tm")
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "data/extremophile_dataset"),
                        help="Destination directory for extremophile datasets")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    csv_dir = output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*70}\nStableProt Extremophile Dataset Curation Pipeline\nTarget Directory: {output_dir}\n{'='*70}")
    
    # 1. Curate OGT Extremophiles
    df_ogt = curate_ogt_benchmarks(output_dir)
    ogt_csv_path = csv_dir / "extremophile_ogt_benchmark.csv"
    df_ogt.to_csv(ogt_csv_path, index=False)
    print(f"  Saved OGT benchmark to {ogt_csv_path}")
    
    # 2. Curate Tm Extremophiles
    df_tm_bench, df_tm_all = curate_tm_benchmarks(output_dir)
    tm_csv_path = csv_dir / "extremophile_tm_benchmark.csv"
    df_tm_bench.to_csv(tm_csv_path, index=False)
    print(f"  Saved Tm benchmark to {tm_csv_path}")
    
    # 3. Curate Paired OGT-Tm
    df_paired = curate_paired_ogt_tm(df_tm_all)
    paired_csv_path = csv_dir / "extremophile_paired_ogt_tm.csv"
    df_paired.to_csv(paired_csv_path, index=False)
    print(f"  Saved Paired OGT-Tm to {paired_csv_path}")
    
    # 4. Master Unified Catalog
    df_master = pd.concat([df_ogt, df_tm_bench], ignore_index=True)
    master_csv_path = csv_dir / "extremophile_master_catalog.csv"
    df_master.to_csv(master_csv_path, index=False)
    print(f"  Saved Master Catalog to {master_csv_path} ({len(df_master)} total records)")
    
    # 5. Export FASTA
    export_fasta_files(df_ogt, df_tm_bench, output_dir)
    
    # 6. Build PyTorch Bundles
    build_pytorch_bundle(df_ogt, df_tm_bench, df_paired, output_dir)
    
    # 7. Generate README
    generate_readme(df_ogt, df_tm_bench, df_paired, output_dir)
    
    print(f"\n{'='*70}\nExtremophile Dataset Curation Complete!\n{'='*70}")

if __name__ == "__main__":
    main()
