import os
import json
import lmdb
import pandas as pd
import glob
import subprocess
from pathlib import Path

def clean_seq(seq):
    # Extracts only amino acid sequence (uppercase letters)
    return "".join([c for c in str(seq) if c.isupper() and c.isalpha()])

def parse_fasta(fasta_path):
    seqs = set()
    if not os.path.exists(fasta_path):
        return seqs
    current_seq = []
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    seqs.add(clean_seq("".join(current_seq)))
                    current_seq = []
            else:
                current_seq.append(line)
        if current_seq:
            seqs.add(clean_seq("".join(current_seq)))
    return seqs

def parse_lmdb(lmdb_path):
    seqs = set()
    if not os.path.exists(lmdb_path):
        return seqs
    env = lmdb.open(lmdb_path, readonly=True)
    with env.begin() as txn:
        cursor = txn.cursor()
        for k, v in cursor:
            try:
                data = json.loads(v.decode('utf-8'))
                if 'seq' in data:
                    seqs.add(clean_seq(data['seq']))
                elif 'seq_1' in data and 'seq_2' in data:
                    seqs.add(clean_seq(data['seq_1']))
                    seqs.add(clean_seq(data['seq_2']))
            except Exception:
                pass
    return seqs

def main():
    print("Collecting benchmark sequences...")
    bench_tasks = {}
    
    # 1. ProteinGym
    pg_path = "data/emergent_benchmarks/DMS_substitutions.parquet"
    if os.path.exists(pg_path):
        df = pd.read_parquet(pg_path)
        bench_tasks["ProteinGym"] = set(clean_seq(s) for s in df["target_seq"].dropna().unique())
    else:
        bench_tasks["ProteinGym"] = set()
        
    # 2. PPI - Yeast
    yeast_ppi_path = "data/emergent_benchmarks/DeepFE-PPI/dataset/11188/protein"
    bench_tasks["PPI_Yeast"] = parse_fasta(yeast_ppi_path)
    
    # 2b. PPI - Human
    human_ppi_test = "data/emergent_benchmarks/HumanPPI/normal/test"
    bench_tasks["PPI_Human"] = parse_lmdb(human_ppi_test)
    
    # 3. DeepLoc
    deeploc_cls2 = "data/emergent_benchmarks/DeepLoc/cls2/normal/test"
    deeploc_cls10 = "data/emergent_benchmarks/DeepLoc/cls10/normal/test"
    bench_tasks["DeepLoc"] = parse_lmdb(deeploc_cls2).union(parse_lmdb(deeploc_cls10))
    
    # 4. eSOL
    esol_test = "data/emergent_benchmarks/eSOL/test.csv"
    if os.path.exists(esol_test):
        df = pd.read_csv(esol_test)
        bench_tasks["eSOL"] = set(clean_seq(s) for s in df["aa_seq"].dropna().unique())
    else:
        bench_tasks["eSOL"] = set()
        
    # 5. EC
    ec_test = "data/emergent_benchmarks/EC/AF2/normal/test"
    bench_tasks["EC"] = parse_lmdb(ec_test)
    
    # 6. CB513
    cb513_path = "data/emergent_benchmarks/CB513/CB513.csv"
    if os.path.exists(cb513_path):
        df = pd.read_csv(cb513_path)
        bench_tasks["CB513"] = set(clean_seq(s) for s in df["input"].dropna().unique())
    else:
        bench_tasks["CB513"] = set()
        
    # 7. SCOP
    scop_test = "data/emergent_benchmarks/scop/test.parquet"
    if os.path.exists(scop_test):
        df = pd.read_parquet(scop_test)
        bench_tasks["SCOP"] = set(clean_seq(s) for s in df["seq"].dropna().unique())
    else:
        bench_tasks["SCOP"] = set()
        
    # 8. LiveProteinBench
    lpb_seqs = set()
    lpb_files = glob.glob("data/emergent_benchmarks/LiveProteinBench/dataset/QA/*.json")
    for f in lpb_files:
        try:
            with open(f, 'r') as fh:
                data = json.load(fh)
                for item in data:
                    if "Protein Sequence" in item:
                        lpb_seqs.add(clean_seq(item["Protein Sequence"]))
        except Exception:
            pass
    bench_tasks["LiveProteinBench"] = lpb_seqs

    # Print summary of benchmark collections
    print("\n--- Benchmark Sequence Counts ---")
    all_bench_seqs = set()
    for task, seqs in bench_tasks.items():
        print(f"{task}: {len(seqs)} sequences")
        all_bench_seqs.update(seqs)
    print(f"Total Unique Benchmark Sequences: {len(all_bench_seqs)}")

    # Collect training sequences
    print("\nCollecting training sequences...")
    train_seqs = set()
    
    # CD-HIT combined output (Tm training)
    train_seqs.update(parse_fasta("data/training_data/raw/cdhit_combined_output.fasta"))
    
    # Meltome sequences if not in CD-HIT
    train_seqs.update(parse_fasta("data/training_data/raw/meltome_sequences.fasta"))
    
    # OGT training clean
    train_seqs.update(parse_fasta("data/training_data/ogt/ogt_training_clean.fasta"))
    train_seqs.update(parse_fasta("data/training_data/ogt/ogt_training_leak_free.fasta"))
    
    # TemStaPro imbalanced training (BacDive OGT)
    train_seqs.update(parse_fasta("data/training_data/temstapro/TemStaPro-Major-30-imbal-training.fasta"))
    
    print(f"Total Unique Training Sequences: {len(train_seqs)}")
    
    # Write to FASTA files
    print("\nWriting sequences to FASTA files...")
    train_fasta = "data/emergent_benchmarks/audit_train.fasta"
    bench_fasta = "data/emergent_benchmarks/audit_bench.fasta"
    
    with open(train_fasta, 'w') as f:
        for i, seq in enumerate(sorted(train_seqs)):
            if len(seq) >= 10:
                f.write(f">train_{i}\n{seq}\n")
                
    # Map benchmark seqs to index for tracking
    bench_list = sorted(all_bench_seqs)
    seq_to_task = {}
    with open(bench_fasta, 'w') as f:
        for i, seq in enumerate(bench_list):
            if len(seq) >= 10:
                f.write(f">bench_{i}\n{seq}\n")
                tasks = [task for task, s_set in bench_tasks.items() if seq in s_set]
                seq_to_task[f"bench_{i}"] = tasks
                
    # Run mmseqs easy-search
    print("\nRunning MMseqs2 easy-search homology leakage audit (threshold = 30% sequence identity, 80% coverage)...")
    audit_out = "data/emergent_benchmarks/audit_out.m8"
    tmp_dir = "data/emergent_benchmarks/tmp_mmseqs"
    os.makedirs(tmp_dir, exist_ok=True)
    
    cmd = [
        "conda", "run", "-n", "stableprot_v2",
        "mmseqs", "easy-search",
        bench_fasta,
        train_fasta,
        audit_out,
        tmp_dir,
        "--min-seq-id", "0.3",
        "-c", "0.8",
        "--search-type", "1"
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("MMseqs2 easy-search failed! Output:")
        print(r.stdout)
        print(r.stderr)
        return
        
    print(f"Parsing MMseqs2 results from {audit_out}...")
    
    leaked_bench_ids = set()
    if os.path.exists(audit_out):
        with open(audit_out, 'r') as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    query_id = parts[0]
                    target_id = parts[1]
                    seq_id_frac = float(parts[2])
                    # MMseqs2 outputs sequence identity as a fraction (0.0 to 1.0)
                    if seq_id_frac >= 0.30:
                        leaked_bench_ids.add(query_id)
                        
    # Calculate leakage metrics per task
    task_leaked = {task: 0 for task in bench_tasks.keys()}
    task_total = {task: len(seqs) for task, seqs in bench_tasks.items()}
    
    for seq_id in leaked_bench_ids:
        if seq_id in seq_to_task:
            for task in seq_to_task[seq_id]:
                task_leaked[task] += 1
                
    # Print beautiful report table
    print("\n" + "="*60)
    print(" HOMOLOGY LEAKAGE AUDIT REPORT (MMseqs2 @ 30% Sequence Identity)")
    print("="*60)
    print(f"{'Benchmark Task':<25} | {'Total Seqs':<10} | {'Leaked Seqs':<12} | {'Leakage %':<10}")
    print("-"*60)
    for task in sorted(bench_tasks.keys()):
        tot = task_total[task]
        leak = task_leaked[task]
        pct = (leak / tot * 100) if tot > 0 else 0.0
        print(f"{task:<25} | {tot:<10} | {leak:<12} | {pct:.2f}%")
    print("="*60)
    
    # Save report to json
    report = {
        "threshold": 0.3,
        "metrics": {
            task: {
                "total": task_total[task],
                "leaked": task_leaked[task],
                "percentage": round(task_leaked[task] / task_total[task] * 100, 2) if task_total[task] > 0 else 0.0
            }
            for task in bench_tasks.keys()
        }
    }
    with open("data/emergent_benchmarks/homology_audit_report.json", 'w') as fh:
        json.dump(report, fh, indent=2)
    print("Homology leakage audit report saved to data/emergent_benchmarks/homology_audit_report.json")

if __name__ == "__main__":
    main()
