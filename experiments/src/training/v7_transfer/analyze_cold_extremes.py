import torch
import numpy as np

def main():
    data_path = "/home/bibhu/Documents/temstampto/data/embeddings/prepared_data_v4.pt"
    print("Loading dataset...")
    dataset = torch.load(data_path, map_location='cpu')
    train_ogt = dataset["train_ogt"]
    sequences = train_ogt["sequences"]
    labels = train_ogt["labels"] if "labels" in train_ogt else train_ogt["ogt_original"]
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    labels = np.array(labels)
    
    print("\nAnalyzing cold-adapted (psychrophilic) proteins (<20°C)...")
    idx = np.where(labels < 20.0)[0]
    print(f"Total psychrophilic sequences: {len(idx)}")
    
    if len(idx) == 0:
        return
        
    cys_pcts = []
    asn_pcts = []
    gln_pcts = []
    lengths = []
    
    for i in idx:
        seq = sequences[i]
        lengths.append(len(seq))
        cys_pcts.append(seq.count('C') / len(seq) * 100)
        asn_pcts.append(seq.count('N') / len(seq) * 100)
        gln_pcts.append(seq.count('Q') / len(seq) * 100)
        
    lengths = np.array(lengths)
    cys_pcts = np.array(cys_pcts)
    asn_pcts = np.array(asn_pcts)
    gln_pcts = np.array(gln_pcts)
    
    print(f"Length: mean={lengths.mean():.1f}, std={lengths.std():.1f}, max={lengths.max()}")
    print(f"Cys %:  mean={cys_pcts.mean():.2f}%, std={cys_pcts.std():.2f}%, max={cys_pcts.max():.2f}%")
    print(f"Asn %:  mean={asn_pcts.mean():.2f}%, std={asn_pcts.std():.2f}%, max={asn_pcts.max():.2f}%")
    print(f"Gln %:  mean={gln_pcts.mean():.2f}%, std={gln_pcts.std():.2f}%, max={gln_pcts.max():.2f}%")
    
    # In psychrophiles, proteins should be flexible, meaning they have normal to high levels of Cys, Asn, Gln.
    # If a protein in <20°C OGT has Cys = 0, Asn < 1%, Gln < 1%, it could be a thermophilic contaminant or misannotated taxid!
    # Let's count how many have extremely low values
    low_cys = np.sum(cys_pcts == 0)
    low_asn = np.sum(asn_pcts < 1.0)
    low_gln = np.sum(gln_pcts < 1.0)
    
    print(f"\nPotential outliers (low flexibility/thermophilic-like) at <20°C:")
    print(f"  Cys == 0:   {low_cys} sequences ({low_cys/len(idx)*100:.2f}%)")
    print(f"  Asn < 1%:   {low_asn} sequences ({low_asn/len(idx)*100:.2f}%)")
    print(f"  Gln < 1%:   {low_gln} sequences ({low_gln/len(idx)*100:.2f}%)")
    
    # Let's check how many sequences have very low values for all three
    rigid_idx = np.where((cys_pcts == 0) & (asn_pcts < 1.5) & (gln_pcts < 1.5))[0]
    print(f"Highly rigid/low-flexibility sequences at <20°C: {len(rigid_idx)} ({len(rigid_idx)/len(idx)*100:.2f}%)")

if __name__ == "__main__":
    main()
