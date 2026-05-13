"""
Generate ProtT5 embeddings for Tm datasets.
Self-contained — no dependency on StableProt/prottrans_models.py.
Compatible with transformers 5.x.
"""

import os
import sys
import torch
import hashlib
from datetime import datetime
from Bio import SeqIO

def generate_prott5_embeddings(fasta_files, cache_dir, max_seq_len=1500):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Collect uncached sequences
    cached = set(os.listdir(cache_dir))
    all_seqs = {}  # id -> seq
    
    for fpath in fasta_files:
        total = 0
        skipped = 0
        for record in SeqIO.parse(fpath, 'fasta'):
            seq = str(record.seq)[:max_seq_len]
            h = hashlib.sha256(seq.encode()).hexdigest()
            fname = f'mean_{h}.pt'
            total += 1
            if fname not in cached:
                all_seqs[record.id] = seq
            else:
                skipped += 1
        print(f"{os.path.basename(fpath)}: {total} total, {skipped} cached, {total-skipped} to generate")
    
    if not all_seqs:
        print("All cached!")
        return
    
    print(f"\nTotal to embed: {len(all_seqs)}")
    print("Loading ProtT5...")
    
    from transformers import T5EncoderModel, AutoTokenizer
    
    model_name = 'Rostlab/prot_t5_xl_half_uniref50-enc'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name).to(device).eval()
    
    print(f"Model loaded on {device}")
    
    # Sort by length for efficient batching
    seq_items = sorted(all_seqs.items(), key=lambda x: len(x[1]), reverse=True)
    
    batch_seqs = []
    batch_ids = []
    batch_lens = []
    done = 0
    start = datetime.now()
    
    for seq_id, seq in seq_items:
        # ProtT5 needs spaces between residues
        spaced = ' '.join(list(seq))
        spaced = spaced.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
        batch_seqs.append(spaced)
        batch_ids.append(seq_id)
        batch_lens.append(len(seq))
        
        cumulative_res = sum(batch_lens)
        
        if len(batch_seqs) >= 32 or cumulative_res >= 4000 or (seq_id == seq_items[-1][0]):
            # Process batch
            encoded = tokenizer(batch_seqs, return_tensors='pt', padding=True, truncation=True, max_length=max_seq_len+2)
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            try:
                with torch.no_grad():
                    output = model(input_ids=input_ids, attention_mask=attention_mask)
                
                for i, sid in enumerate(batch_ids):
                    slen = batch_lens[i]
                    emb = output.last_hidden_state[i, :slen]  # exclude padding
                    mean_emb = emb.mean(dim=0).cpu()
                    
                    # Save in same format as original cache
                    seq_raw = all_seqs[sid]
                    h = hashlib.sha256(seq_raw.encode()).hexdigest()
                    save_data = {
                        'label': sid,
                        'sequence': seq_raw,
                        'mean_representations': mean_emb,
                    }
                    torch.save(save_data, os.path.join(cache_dir, f'mean_{h}.pt'))
                    done += 1
                    
            except RuntimeError as e:
                print(f"OOM on batch of {len(batch_seqs)} seqs (max len {max(batch_lens)}). Skipping.")
                print(f"Error: {e}")
            
            batch_seqs = []
            batch_ids = []
            batch_lens = []
            
            if done % 500 == 0 and done > 0:
                elapsed = (datetime.now() - start).total_seconds()
                rate = done / elapsed
                remaining = (len(all_seqs) - done) / rate / 60
                print(f"  {done}/{len(all_seqs)} done ({rate:.1f} seq/s, ~{remaining:.0f} min remaining)")
    
    elapsed = (datetime.now() - start).total_seconds()
    print(f"\nDone! {done} embeddings in {elapsed:.0f}s ({done/elapsed:.1f} seq/s)")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'embeddings_cache')
    
    fasta_files = [
        os.path.join(project_root, 'new_data', 'meltome_sequences.fasta'),
        os.path.join(project_root, 'new_data', 'prothermdb_validation.fasta'),
        os.path.join(project_root, 'new_data', 'tembert_test_sequences.fasta'),
    ]
    fasta_files = [f for f in fasta_files if os.path.exists(f)]
    
    generate_prott5_embeddings(fasta_files, cache_dir)
