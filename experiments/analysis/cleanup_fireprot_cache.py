import torch
import hashlib
import os

def main():
    p = 'experiments/data_processing/fireprot_holdout_prott5.pt'
    if not os.path.exists(p):
        print(f"File {p} not found.")
        return
        
    d = torch.load(p, map_location='cpu')
    cache_dir = 'experiments/esm2_embeddings_cache'
    deleted = 0
    
    for seq in d['sequences']:
        h = hashlib.sha256(seq[:1500].encode()).hexdigest()
        f = os.path.join(cache_dir, f'esm2_{h}.pt')
        if os.path.exists(f):
            os.remove(f)
            deleted += 1
            
    print(f"Deleted {deleted} files from {cache_dir}.")

if __name__ == "__main__":
    main()
