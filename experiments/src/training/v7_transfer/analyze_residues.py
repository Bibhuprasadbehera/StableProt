import torch
import numpy as np

def main():
    data_path = "/home/bibhu/Documents/temstampto/data/embeddings/prepared_data_v4.pt"
    print("Loading dataset...")
    dataset = torch.load(data_path, map_location='cpu')
    train_ogt = dataset["train_ogt"]
    
    # 1. Inspect chaperone_client
    cc = train_ogt.get("chaperone_client")
    if cc is not None:
        if isinstance(cc, torch.Tensor):
            cc_np = cc.numpy()
        else:
            cc_np = np.array(cc)
        print(f"\nChaperone client info present. Type: {type(cc)}")
        print(f"Total sequences: {len(cc_np)}")
        print(f"Non-zero elements: {np.count_nonzero(cc_np)}")
        print(f"Unique values in chaperone_client: {np.unique(cc_np)}")
    else:
        print("\nNo chaperone_client key found.")
        
    # 2. Inspect ogt_reliable
    rel = train_ogt.get("ogt_reliable")
    if rel is not None:
        if isinstance(rel, torch.Tensor):
            rel_np = rel.numpy()
        else:
            rel_np = np.array(rel)
        print(f"\nReliable info present. Unique values: {np.unique(rel_np)}")
        print(f"Number of reliable sequences: {np.sum(rel_np == 1)} ({np.mean(rel_np == 1)*100:.2f}%)")
        
    # 3. Analyze residue content vs OGT
    sequences = train_ogt["sequences"]
    labels = train_ogt["labels"] if "labels" in train_ogt else train_ogt["ogt_original"]
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    labels = np.array(labels)
    
    print("\nAnalyzing amino acid composition vs temperature...")
    bins = [(0, 30), (30, 45), (45, 60), (60, 80), (80, 110)]
    
    for low, high in bins:
        idx = np.where((labels >= low) & (labels < high))[0]
        if len(idx) == 0:
            continue
            
        # Sample up to 5000 sequences to make analysis fast
        sample_size = min(5000, len(idx))
        sampled_idx = np.random.choice(idx, sample_size, replace=False)
        
        cys_pcts = []
        asn_pcts = []
        gln_pcts = []
        lengths = []
        
        for i in sampled_idx:
            seq = sequences[i]
            lengths.append(len(seq))
            cys_pcts.append(seq.count('C') / len(seq) * 100)
            asn_pcts.append(seq.count('N') / len(seq) * 100)
            gln_pcts.append(seq.count('Q') / len(seq) * 100)
            
        print(f"OGT Bin [{low}-{high}°C] (n={len(idx)}):")
        print(f"  Mean Length: {np.mean(lengths):.1f} aa")
        print(f"  Mean Cys %:  {np.mean(cys_pcts):.2f}%")
        print(f"  Mean Asn %:  {np.mean(asn_pcts):.2f}%")
        print(f"  Mean Gln %:  {np.mean(gln_pcts):.2f}%")
        
if __name__ == "__main__":
    main()
