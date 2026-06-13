"""
Retry ESMFold structure generation for OOM sequences.
Strategy: Truncate long sequences to fit GPU memory.
- Sequences 871-2000 aa, all failed at 1000-truncation
- Try progressively shorter truncation: 800 → 600 → 400
- Use torch.cuda.empty_cache() between batches
- Enable fp16/chunking if available
"""
import os
import sys
import time
import json
import torch
import numpy as np
from transformers import AutoTokenizer, EsmForProteinFolding
from tqdm import tqdm

PROJECT_ROOT = "/home/bibhu/Documents/temstampto"

def convert_outputs_to_pdb(outputs, sequence):
    """Convert ESMFold output to PDB format string."""
    positions = outputs.positions[-1]  # Final layer, [1, L, 37, 3]
    positions = positions[0].cpu().numpy()  # [L, 37, 3]
    
    atom_names = ["N", "CA", "C", "O", "CB"]
    mapping = {
        'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
        'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
        'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
        'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
    }
    
    lines = []
    atom_idx = 1
    for res_idx, aa in enumerate(sequence):
        for atom_i, atom_name in enumerate(atom_names):
            if atom_i >= positions.shape[1]:
                break
            x, y, z = positions[res_idx, atom_i]
            if np.isnan(x):
                continue
            three = mapping.get(aa.upper(), 'UNK')
            lines.append(
                f"ATOM  {atom_idx:5d} {atom_name:4s} {three} A{res_idx+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_name[0]:>2s}"
            )
            atom_idx += 1
    lines.append("END")
    return "\n".join(lines)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load dataset
    data_path = f"{PROJECT_ROOT}/data/embeddings/prepared_data_v4_saprot_cleaned.pt"
    print(f"Loading dataset...")
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    
    # Load error list
    out_dir = f"{PROJECT_ROOT}/data/structures/tm_esmfold"
    error_path = os.path.join(out_dir, "errors.json")
    with open(error_path) as f:
        errors = json.load(f)
    
    error_ids = set(e[0] for e in errors)
    print(f"Sequences to retry: {len(error_ids)}")
    
    # Build work list with sequences
    work = []
    for seq_id in error_ids:
        parts = seq_id.rsplit('_', 1)
        split_name = parts[0]
        idx = int(parts[1])
        if split_name in data and 'sequences' in data[split_name]:
            seq = str(data[split_name]['sequences'][idx])
            work.append((seq_id, seq))
    
    # Sort by length (shortest first for early wins)
    work.sort(key=lambda x: len(x[1]))
    print(f"Length range: {len(work[0][1])} - {len(work[-1][1])}")
    
    # Load ESMFold 
    print("Loading ESMFold model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    model.to(device)
    model.eval()
    
    # Enable memory-efficient settings
    if hasattr(model, 'esm'):
        model.esm.float()  # ESM encoder in fp32 for stability
    # Set trunk chunk_size for reduced memory
    if hasattr(model, 'set_chunk_size'):
        model.set_chunk_size(64)  # Process attention in chunks
    
    print(f"Model loaded in {time.time() - t0:.1f}s")
    
    # Warmup
    with torch.no_grad():
        dummy = tokenizer(["MAST"], return_tensors="pt", add_special_tokens=False)
        dummy = {k: v.to(device) for k, v in dummy.items()}
        _ = model(**dummy)
    
    pdb_dir = os.path.join(out_dir, "pdbs")
    progress_path = os.path.join(out_dir, "progress.json")
    
    # Load existing progress
    with open(progress_path) as f:
        progress = json.load(f)
    completed_set = set(progress) if isinstance(progress, list) else set(progress.get("completed", progress))
    
    # Truncation levels to try
    MAX_LENGTHS = [900, 800, 700, 600]
    
    new_completed = 0
    new_errors = []
    save_interval = 20
    
    pbar = tqdm(work, desc="ESMFold retry", unit="seq")
    for idx, (seq_id, seq) in enumerate(pbar):
        if seq_id in completed_set:
            continue
            
        success = False
        for max_len in MAX_LENGTHS:
            seq_trunc = seq[:max_len]
            try:
                torch.cuda.empty_cache()
                with torch.no_grad():
                    inputs = tokenizer([seq_trunc], return_tensors="pt", add_special_tokens=False)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = model(**inputs)
                
                pdb_str = convert_outputs_to_pdb(outputs, seq_trunc)
                pdb_path = os.path.join(pdb_dir, f"{seq_id}.pdb")
                with open(pdb_path, 'w') as f:
                    f.write(pdb_str)
                
                completed_set.add(seq_id)
                if isinstance(progress, list):
                    progress.append(seq_id)
                else:
                    progress.setdefault("completed", []).append(seq_id)
                new_completed += 1
                success = True
                
                pbar.set_postfix({
                    'done': new_completed,
                    'len': len(seq),
                    'trunc': max_len,
                    'fail': len(new_errors)
                })
                break  # Success at this truncation level
                
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                new_errors.append((seq_id, str(e)))
                pbar.set_postfix({'error': str(e)[:30]})
                break
        
        if not success and seq_id not in completed_set:
            new_errors.append((seq_id, f"OOM even at {MAX_LENGTHS[-1]} truncation"))
        
        # Save progress
        if (idx + 1) % save_interval == 0:
            with open(progress_path, 'w') as f:
                json.dump(progress, f)
    
    # Final save
    with open(progress_path, 'w') as f:
        json.dump(progress, f)
    
    if new_errors:
        retry_error_path = os.path.join(out_dir, "retry_errors.json")
        with open(retry_error_path, 'w') as f:
            json.dump(new_errors, f, indent=2)
        print(f"\n{len(new_errors)} retry errors saved to {retry_error_path}")
    
    total_pdbs = len(os.listdir(pdb_dir))
    print(f"\nRetry complete! {new_completed} new structures generated.")
    print(f"Total PDBs now: {total_pdbs}")

if __name__ == "__main__":
    main()
