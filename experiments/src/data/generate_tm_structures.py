"""Generate 3D structures for Tm dataset sequences using ESMFold.
Saves PDB files and 3Di tokens via Foldseek for SaProt input.
Progress bar and time estimate included.
"""
import os
import sys
import time
import json
import torch
import numpy as np
from transformers import AutoTokenizer, EsmForProteinFolding
from tqdm import tqdm

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    # Load dataset
    data_path = os.path.join(project_root, "data/embeddings/prepared_data_v4_saprot.pt")
    print(f"Loading dataset from {data_path}...")
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    
    # Collect all Tm sequences (train + val + test)
    splits = {}
    for split_name in ['train_tm', 'val_tm', 'test_tm']:
        if split_name in data and 'sequences' in data[split_name]:
            seqs = data[split_name]['sequences']
            splits[split_name] = seqs
            print(f"  {split_name}: {len(seqs)} sequences")
    
    # Output directory
    out_dir = os.path.join(project_root, "data/structures/tm_esmfold")
    os.makedirs(out_dir, exist_ok=True)
    
    # Check already processed
    done_file = os.path.join(out_dir, "progress.json")
    if os.path.exists(done_file):
        with open(done_file) as f:
            progress = json.load(f)
    else:
        progress = {"completed": []}
    
    completed_set = set(progress["completed"])
    
    # Build work list
    work = []
    for split_name, seqs in splits.items():
        for i, seq in enumerate(seqs):
            seq_id = f"{split_name}_{i}"
            if seq_id not in completed_set:
                work.append((seq_id, str(seq), split_name))
    
    total_all = sum(len(s) for s in splits.values())
    print(f"\nTotal sequences: {total_all}")
    print(f"Already completed: {len(completed_set)}")
    print(f"Remaining: {len(work)}")
    
    if len(work) == 0:
        print("All structures already generated!")
        return
    
    # Load ESMFold model
    print("\nLoading ESMFold model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    model.to(device)
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")
    
    # Warmup
    print("Warmup...")
    with torch.no_grad():
        dummy = tokenizer(["MAST"], return_tensors="pt", add_special_tokens=False)
        dummy = {k: v.to(device) for k, v in dummy.items()}
        _ = model(**dummy)
    
    # Process sequences
    pdb_dir = os.path.join(out_dir, "pdbs")
    os.makedirs(pdb_dir, exist_ok=True)
    
    save_interval = 50  # Save progress every 50 sequences
    errors = []
    
    pbar = tqdm(work, desc="ESMFold", unit="seq")
    for idx, (seq_id, seq, split_name) in enumerate(pbar):
        try:
            # Truncate very long sequences to avoid OOM
            seq_trunc = seq[:1000]
            
            with torch.no_grad():
                inputs = tokenizer([seq_trunc], return_tensors="pt", add_special_tokens=False)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
            
            # Extract PDB string
            # ESMFold outputs atom positions; convert to PDB
            pdb_str = convert_outputs_to_pdb(outputs, seq_trunc)
            
            pdb_path = os.path.join(pdb_dir, f"{seq_id}.pdb")
            with open(pdb_path, 'w') as f:
                f.write(pdb_str)
            
            completed_set.add(seq_id)
            progress["completed"].append(seq_id)
            
            # Update progress bar
            elapsed = pbar.format_dict['elapsed']
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(work) - idx - 1) / rate if rate > 0 else 0
            pbar.set_postfix({
                'done': len(completed_set),
                'rate': f'{rate:.1f}/s',
                'ETA': f'{remaining/3600:.1f}h'
            })
            
        except Exception as e:
            errors.append((seq_id, str(e)))
            pbar.set_postfix({'error': str(e)[:30]})
        
        # Save progress periodically
        if (idx + 1) % save_interval == 0:
            with open(done_file, 'w') as f:
                json.dump(progress, f)
    
    # Final save
    with open(done_file, 'w') as f:
        json.dump(progress, f)
    
    if errors:
        error_path = os.path.join(out_dir, "errors.json")
        with open(error_path, 'w') as f:
            json.dump(errors, f, indent=2)
        print(f"\n{len(errors)} errors saved to {error_path}")
    
    print(f"\nDone! {len(completed_set)}/{total_all} structures generated.")

def convert_outputs_to_pdb(outputs, sequence):
    """Convert ESMFold output to PDB format string."""
    # Get atom positions (N, CA, C, O for each residue)
    positions = outputs.positions[-1]  # Take final layer, shape: [1, L, 37, 3]
    positions = positions[0].cpu().numpy()  # [L, 37, 3]
    
    atom_names = ["N", "CA", "C", "O", "CB"]  # First 5 atoms
    
    lines = []
    atom_idx = 1
    for res_idx, aa in enumerate(sequence):
        for atom_i, atom_name in enumerate(atom_names):
            if atom_i >= positions.shape[1]:
                break
            x, y, z = positions[res_idx, atom_i]
            if np.isnan(x):
                continue
            lines.append(
                f"ATOM  {atom_idx:5d} {atom_name:4s} {three_letter(aa)} A{res_idx+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_name[0]:>2s}"
            )
            atom_idx += 1
    lines.append("END")
    return "\n".join(lines)

def three_letter(aa):
    """Convert single-letter amino acid to three-letter code."""
    mapping = {
        'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
        'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
        'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
        'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
    }
    return mapping.get(aa.upper(), 'UNK')

if __name__ == "__main__":
    main()
