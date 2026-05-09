import os
import argparse
import torch
import esm
import gc
from tqdm import tqdm

def read_fasta_subset(file_path: str, max_count: int = 1000):
    sequences = []
    targets = []
    with open(file_path, 'r') as f:
        seq_id = ""
        seq = ""
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id:
                    # Parse target from header (TemStaPro format: >id|target)
                    # Example: >tr|A0A024RBG1|A0A024RBG1_HUMAN|37.0
                    parts = seq_id.split('|')
                    if len(parts) > 1:
                        try:
                            # It could be the last part
                            target = float(parts[-1])
                        except ValueError:
                            # Try something else or default
                            target = 37.0
                    else:
                        target = 37.0
                        
                    sequences.append((seq_id, seq))
                    targets.append(target)
                    
                    if len(sequences) >= max_count:
                        break
                        
                seq_id = line[1:]
                seq = ""
            else:
                seq += line
        if seq_id and len(sequences) < max_count:
            parts = seq_id.split('|')
            if len(parts) > 1:
                try:
                    target = float(parts[-1])
                except ValueError:
                    target = 37.0
            else:
                target = 37.0
            sequences.append((seq_id, seq))
            targets.append(target)
            
    return sequences, targets

def main():
    fasta_path = "../dataset/TemStaPro-Major-30-imbal-training.fasta"
    cache_dir = "esm2_embeddings_cache/subset_1k"
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"Reading subset from {fasta_path}...")
    sequences, targets = read_fasta_subset(fasta_path, 1000)
    print(f"Loaded {len(sequences)} sequences.")
    
    # Save targets
    torch.save(targets, os.path.join(cache_dir, "targets.pt"))
    print("Saved targets.pt")
    
    print("Loading ESM-2 3B model...")
    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    batch_converter = alphabet.get_batch_converter()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.eval().to(device)
    
    layers = [30, 33, 36]
    batch_size = 2  # Lowered from 8 to avoid OOM
    
    # Sort by length
    # Also truncate to 1500 just in case
    seqs_with_targets = [( (s[0], s[1][:1500]), t ) for s, t in zip(sequences, targets)]
    seqs_with_targets.sort(key=lambda x: len(x[0][1]))
    
    total_processed = 0
    
    with torch.no_grad():
        for i in tqdm(range(0, len(seqs_with_targets), batch_size)):
            batch = seqs_with_targets[i:i+batch_size]
            data = [item[0] for item in batch] # [(id, seq), ...]
            
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)
            
            results = model(batch_tokens, repr_layers=layers, return_contacts=False)
            
            for j, ((sid, seq), target) in enumerate(batch):
                out_path = os.path.join(cache_dir, f"esm2_subset_{i+j}.pt")
                seq_len = len(seq)
                
                layer_reps = []
                for layer in layers:
                    token_reps = results["representations"][layer][j, 1:seq_len+1]
                    mean_rep = token_reps.mean(0).cpu()
                    layer_reps.append(mean_rep)
                    
                stacked_reps = torch.stack(layer_reps) # (3, 2560)
                torch.save(stacked_reps, out_path)
                total_processed += 1
                
            del batch_tokens, results
            torch.cuda.empty_cache()
            
    print(f"Generated embeddings for {total_processed} sequences.")

if __name__ == "__main__":
    main()
