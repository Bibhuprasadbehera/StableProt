#!/usr/bin/env python3
"""
Phase 2: SaProt 1.3B Embedding Generation (Sequence-Only Mode)

Generates frozen SaProt 1.3B embeddings for ALL sequences using sequence-only mode.
No structures needed — structure tokens masked with '#'.

Usage:
    python phase2_saprot_embeddings.py --dataset tm   [--batch-size 8]
    python phase2_saprot_embeddings.py --dataset ogt  [--batch-size 8]
    python phase2_saprot_embeddings.py --dataset eval [--batch-size 8]
    python phase2_saprot_embeddings.py --merge  # Combine all into final .pt file

Run on HPC with GPU. Each dataset can be processed independently.
"""

import argparse
import os
import gc
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPLITS_FILE = PROJECT_ROOT / "data" / "embeddings" / "prepared_data_v7_splits.pt"
OUTPUT_DIR = PROJECT_ROOT / "data" / "embeddings" / "saprot_1.3b"
FINAL_OUTPUT = PROJECT_ROOT / "data" / "embeddings" / "prepared_data_v7_saprot1.3b_seqonly.pt"

MODEL_NAME = "westlake-repl/SaProt_1.3B_AFDB_OMG_NCBI"
EMBEDDING_DIM = 1280


def mask_sequence_for_saprot(seq: str) -> str:
    """Convert amino acid sequence to SaProt sequence-only format.
    
    SaProt expects alternating AA + structure token pairs.
    In sequence-only mode, structure tokens are masked with '#'.
    Example: 'MVLS' -> 'M#V#L#S#'
    """
    return "".join(f"{aa}#" for aa in seq)


def load_saprot_model(device="cuda"):
    """Load frozen SaProt 1.3B model and tokenizer."""
    from transformers import EsmTokenizer, EsmModel

    print(f"Loading {MODEL_NAME}...")
    tokenizer = EsmTokenizer.from_pretrained(MODEL_NAME)
    model = EsmModel.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {total_params / 1e6:.0f}M parameters")
    print(f"  Device: {device}")

    return model, tokenizer


def extract_embeddings_batched(model, tokenizer, sequences, batch_size=8, device="cuda", max_length=1024):
    """Extract mean-pooled embeddings from SaProt in batches.
    
    Uses length-sorting / dynamic padding to maximize speed.
    Returns: torch.Tensor of shape (N, EMBEDDING_DIM)
    """
    # Sort by length to minimize padding overhead
    sorted_indices = sorted(range(len(sequences)), key=lambda idx: len(sequences[idx]))
    
    # Pre-allocate output on CPU
    all_embeddings = torch.zeros((len(sequences), EMBEDDING_DIM), dtype=torch.float32)

    for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embeddings"):
        batch_indices = sorted_indices[i:i + batch_size]
        batch_seqs = [sequences[idx] for idx in batch_indices]

        # Convert to SaProt format (sequence-only with # mask)
        saprot_seqs = [mask_sequence_for_saprot(seq) for seq in batch_seqs]

        # Truncate if needed
        saprot_seqs = [s[:max_length * 2] for s in saprot_seqs]

        # Tokenize
        inputs = tokenizer(
            saprot_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        # Forward pass
        with torch.no_grad(), torch.amp.autocast('cuda'):
            outputs = model(**inputs)
            # Mean pool over sequence length (excluding padding)
            attention_mask = inputs["attention_mask"].unsqueeze(-1)  # (B, L, 1)
            hidden = outputs.last_hidden_state  # (B, L, EMBEDDING_DIM)
            masked_hidden = hidden * attention_mask
            embeddings = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)  # (B, EMBEDDING_DIM)

        all_embeddings[batch_indices] = embeddings.cpu().float()

        # Free GPU memory periodically
        if (i // batch_size) % 100 == 0:
            torch.cuda.empty_cache()

    return all_embeddings


def process_tm_dataset(model, tokenizer, device, batch_size):
    """Process Tm train/val/test splits."""
    print("\n=== Processing Tm Dataset ===")
    data = torch.load(SPLITS_FILE, map_location='cpu', weights_only=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in ['train_tm', 'val_tm', 'test_tm']:
        out_path = OUTPUT_DIR / f"{split}_embeddings.pt"
        if out_path.exists():
            print(f"  {split}: already exists, skipping")
            continue

        sequences = data[split]['sequences']
        print(f"\n  {split}: {len(sequences)} sequences")

        embeddings = extract_embeddings_batched(
            model, tokenizer, sequences,
            batch_size=batch_size, device=device
        )
        print(f"  Embeddings shape: {embeddings.shape}")

        torch.save(embeddings, out_path)
        print(f"  Saved to {out_path}")

    del data
    gc.collect()


def process_ogt_dataset(model, tokenizer, device, batch_size):
    """Process OGT training sequences."""
    print("\n=== Processing OGT Dataset ===")
    data = torch.load(SPLITS_FILE, map_location='cpu', weights_only=False)
    sequences = data['train_ogt']['sequences']
    print(f"  OGT sequences: {len(sequences)}")

    out_path = OUTPUT_DIR / "train_ogt_embeddings.pt"
    if out_path.exists():
        print(f"  Already exists, skipping")
        del data
        gc.collect()
        return

    # Process in chunks to manage memory (941K sequences)
    CHUNK_SIZE = 50000
    all_embeddings = []

    for chunk_start in range(0, len(sequences), CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, len(sequences))
        chunk_seqs = sequences[chunk_start:chunk_end]
        print(f"\n  Chunk {chunk_start//CHUNK_SIZE + 1}: sequences {chunk_start}-{chunk_end}")

        checkpoint_path = OUTPUT_DIR / f"ogt_chunk_{chunk_start}.pt"
        if checkpoint_path.exists():
            print(f"  Loading checkpoint from {checkpoint_path}...")
            chunk_emb = torch.load(checkpoint_path, map_location='cpu')
        else:
            chunk_emb = extract_embeddings_batched(
                model, tokenizer, chunk_seqs,
                batch_size=batch_size, device=device
            )
            torch.save(chunk_emb, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")
            
        all_embeddings.append(chunk_emb)

    embeddings = torch.cat(all_embeddings, dim=0)
    print(f"\n  Final OGT embeddings shape: {embeddings.shape}")

    torch.save(embeddings, out_path)
    print(f"  Saved to {out_path}")

    # Clean up chunk files
    for chunk_start in range(0, len(sequences), CHUNK_SIZE):
        checkpoint_path = OUTPUT_DIR / f"ogt_chunk_{chunk_start}.pt"
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    del data, all_embeddings
    gc.collect()


def process_eval_datasets(model, tokenizer, device, batch_size):
    """Process ProThermDB and FireProtDB evaluation sequences."""
    print("\n=== Processing Evaluation Datasets ===")

    eval_files = {
        'protherm': PROJECT_ROOT / "data" / "embeddings" / "protherm_eval_data.pt",
        'fireprot': PROJECT_ROOT / "data" / "embeddings" / "fireprot_eval_data.pt",
    }

    for name, path in eval_files.items():
        out_path = OUTPUT_DIR / f"{name}_embeddings.pt"
        if out_path.exists():
            print(f"  {name}: already exists, skipping")
            continue

        if not path.exists():
            print(f"  {name}: source file not found at {path}, skipping")
            continue

        data = torch.load(path, map_location='cpu', weights_only=False)
        sequences = data.get('sequences', [])
        if not sequences:
            print(f"  {name}: no sequences found, skipping")
            continue

        print(f"\n  {name}: {len(sequences)} sequences")
        embeddings = extract_embeddings_batched(
            model, tokenizer, sequences,
            batch_size=batch_size, device=device
        )
        print(f"  Embeddings shape: {embeddings.shape}")

        torch.save(embeddings, out_path)
        print(f"  Saved to {out_path}")


def merge_all():
    """Merge all embeddings into final prepared data file."""
    print("\n=== Merging All Embeddings ===")

    # Load original splits (for metadata)
    data = torch.load(SPLITS_FILE, map_location='cpu', weights_only=False)

    # Replace embeddings with SaProt 1.3B
    for split in ['train_tm', 'val_tm', 'test_tm']:
        emb_path = OUTPUT_DIR / f"{split}_embeddings.pt"
        if emb_path.exists():
            new_emb = torch.load(emb_path, map_location='cpu', weights_only=False)
            data[split]['embeddings'] = new_emb
            print(f"  {split}: replaced embeddings → {new_emb.shape}")
        else:
            print(f"  WARNING: {emb_path} not found!")

    # OGT embeddings
    ogt_emb_path = OUTPUT_DIR / "train_ogt_embeddings.pt"
    if ogt_emb_path.exists():
        new_ogt_emb = torch.load(ogt_emb_path, map_location='cpu', weights_only=False)
        data['train_ogt']['embeddings'] = new_ogt_emb
        print(f"  train_ogt: replaced embeddings → {new_ogt_emb.shape}")

    # Save final file
    torch.save(data, FINAL_OUTPUT)
    print(f"\n  Final file saved: {FINAL_OUTPUT}")
    print(f"  File size: {FINAL_OUTPUT.stat().st_size / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Phase 2: SaProt 1.3B Embedding Generation")
    parser.add_argument("--dataset", choices=["tm", "ogt", "eval"], help="Which dataset to process")
    parser.add_argument("--merge", action="store_true", help="Merge all embeddings into final file")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.merge:
        merge_all()
        return

    if not args.dataset:
        parser.error("Must specify --dataset or --merge")

    # Load model once
    model, tokenizer = load_saprot_model(device=args.device)

    if args.dataset == "tm":
        process_tm_dataset(model, tokenizer, args.device, args.batch_size)
    elif args.dataset == "ogt":
        process_ogt_dataset(model, tokenizer, args.device, args.batch_size)
    elif args.dataset == "eval":
        process_eval_datasets(model, tokenizer, args.device, args.batch_size)


if __name__ == "__main__":
    main()
