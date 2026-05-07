import os
import argparse
import hashlib
import torch
import esm
import gc
from typing import List, Tuple
from collections import defaultdict
from tqdm import tqdm

def read_fasta(file_path: str) -> List[Tuple[str, str]]:
    sequences = []
    with open(file_path, 'r') as f:
        seq_id = ""
        seq = ""
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id:
                    sequences.append((seq_id, seq))
                seq_id = line[1:]
                seq = ""
            else:
                seq += line
        if seq_id:
            sequences.append((seq_id, seq))
    return sequences

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta', type=str, required=True, help='Path to FASTA file')
    parser.add_argument('--cache-dir', type=str, default='esm2_embeddings_cache', help='Directory to save embeddings')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size (in sequences)')
    parser.add_argument('--max-seq-len', type=int, default=1500, help='Maximum sequence length')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--layers', type=int, nargs='+', default=[36], help='Layer(s) to extract embeddings from (e.g., 30 33 36). If multiple, output shape is (num_layers, D). If single, output shape is (D,).')
    parser.add_argument('--model-size', type=str, default='3B', choices=['650M', '3B'], help='Model size')
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    print(f"Loading ESM-2 {args.model_size} model...")
    if args.model_size == '3B':
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    else:
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    model = model.to(args.device)
    print("Model loaded.")

    print(f"Reading {args.fasta}...")
    sequences = read_fasta(args.fasta)
    print(f"Found {len(sequences)} sequences.")

    # Filter sequences by length (truncate if too long instead of dropping maybe? No, let's truncate to max_seq_len)
    sequences = [(sid, seq[:args.max_seq_len]) for sid, seq in sequences]
    print(f"Processing {len(sequences)} sequences (truncated to <= {args.max_seq_len}).")

    # Group by length for efficient batching
    seqs_by_length = defaultdict(list)
    for sid, seq in sequences:
        seqs_by_length[len(seq)].append((sid, seq))
    
    batches = []
    for length in sorted(seqs_by_length.keys()):
        group = seqs_by_length[length]
        for i in range(0, len(group), args.batch_size):
            batches.append(group[i:i + args.batch_size])
    
    print(f"Created {len(batches)} batches.")

    total_processed = 0
    total_skipped = 0

    with torch.no_grad():
        for batch in tqdm(batches):
            # Check cache
            batch_to_process = []
            for sid, seq in batch:
                seq_hash = hashlib.sha256(seq.encode()).hexdigest()
                out_path = os.path.join(args.cache_dir, f"esm2_{seq_hash}.pt")
                if not os.path.exists(out_path):
                    batch_to_process.append((sid, seq, out_path))
                else:
                    total_skipped += 1
            
            if not batch_to_process:
                continue

            # Format input
            data = [(sid, seq) for sid, seq, _ in batch_to_process]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
            
            # Forward pass
            batch_tokens = batch_tokens.to(args.device)
            results = model(batch_tokens, repr_layers=args.layers, return_contacts=False)
            
            # Extract and save
            for i, (sid, seq, out_path) in enumerate(batch_to_process):
                seq_len = len(seq)
                if len(args.layers) == 1:
                    layer = args.layers[0]
                    # Get sequence representations (shape: seq_len, embed_dim)
                    token_reps = results["representations"][layer][i, 1:seq_len+1] 
                    # Mean pooling
                    mean_rep = token_reps.mean(0).cpu()
                    torch.save(mean_rep, out_path)
                else:
                    # Save multiple layers
                    layer_reps = []
                    for layer in args.layers:
                        token_reps = results["representations"][layer][i, 1:seq_len+1]
                        mean_rep = token_reps.mean(0).cpu()
                        layer_reps.append(mean_rep)
                    stacked_reps = torch.stack(layer_reps) # shape: (num_layers, embed_dim)
                    torch.save(stacked_reps, out_path)
                
                total_processed += 1
            
            del batch_tokens, results
            torch.cuda.empty_cache()

    print(f"Done! Processed: {total_processed}, Skipped (cached): {total_skipped}")

if __name__ == '__main__':
    main()
