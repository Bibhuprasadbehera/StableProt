import os
import sys
import torch

def purge_embeddings():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "..", "new_data")
    exp_dir = os.path.join(base_dir, "data_processing")
    
    prepared_pt = os.path.join(data_dir, "prepared_data_v2.pt")
    out_pt = os.path.join(data_dir, "prepared_data_v2_leak_free.pt")
    
    if not os.path.exists(prepared_pt):
        print(f"File not found: {prepared_pt}")
        return
        
    print("Loading FireProt holdout sequences...")
    fireprot_data = torch.load(os.path.join(exp_dir, "fireprot_holdout_prott5.pt"), map_location='cpu', weights_only=False)
    forbidden_seqs = set(fireprot_data['sequences'])
    
    print(f"Loading {prepared_pt} (This may take a minute)...")
    data = torch.load(prepared_pt, map_location='cpu', weights_only=False)
    
    # We need to filter train_tm
    if 'train_tm' in data and 'sequences' in data['train_tm']:
        seqs = data['train_tm']['sequences']
        
        # Find indices to keep
        keep_indices = [i for i, seq in enumerate(seqs) if seq not in forbidden_seqs]
        
        dropped = len(seqs) - len(keep_indices)
        print(f"Found {dropped} leaked sequences in Train Tm embeddings.")
        
        if dropped > 0:
            # Filter the lists/tensors
            data['train_tm']['sequences'] = [seqs[i] for i in keep_indices]
            data['train_tm']['labels'] = data['train_tm']['labels'][keep_indices]
            data['train_tm']['embeddings'] = data['train_tm']['embeddings'][keep_indices]
            
            print("Filtering complete. Saving new dictionary...")
            torch.save(data, out_pt)
            print(f"Saved leak-free embeddings to {out_pt}")
        else:
            print("No leakage found in dictionary. Skipping save.")
    else:
        print("Could not find 'train_tm' sequences in dictionary.")

if __name__ == "__main__":
    purge_embeddings()
