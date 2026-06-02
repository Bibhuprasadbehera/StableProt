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
    
    print("\nAnalyzing extreme temperature proteins (>80°C)...")
    idx = np.where(labels >= 80.0)[0]
    print(f"Total hyperthermophilic sequences: {len(idx)}")
    
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
    
    # Let's count outliers
    cys_outliers = np.sum(cys_pcts > 3.0)
    asn_outliers = np.sum(asn_pcts > 7.0)
    gln_outliers = np.sum(gln_pcts > 5.0)
    long_outliers = np.sum(lengths > 800)
    
    print(f"\nPotential outliers at >= 80°C:")
    print(f"  Cys > 3%:   {cys_outliers} sequences ({cys_outliers/len(idx)*100:.2f}%)")
    print(f"  Asn > 7%:   {asn_outliers} sequences ({asn_outliers/len(idx)*100:.2f}%)")
    print(f"  Gln > 5%:   {gln_outliers} sequences ({gln_outliers/len(idx)*100:.2f}%)")
    print(f"  Len > 800:  {long_outliers} sequences ({long_outliers/len(idx)*100:.2f}%)")
    
    # Combined outliers: any sequence having Cys > 3% OR Asn > 7% OR Gln > 5% OR Len > 800
    outliers_idx = np.where((cys_pcts > 3.0) | (asn_pcts > 7.0) | (gln_pcts > 5.0) | (lengths > 800))[0]
    print(f"\nTotal outliers to clean at >= 80°C: {len(outliers_idx)} sequences ({len(outliers_idx)/len(idx)*100:.2f}%)")
    
    # Let's check 60-80°C bin
    print("\nAnalyzing thermophilic proteins (60-80°C)...")
    idx_60 = np.where((labels >= 60.0) & (labels < 80.0))[0]
    print(f"Total thermophilic sequences: {len(idx_60)}")
    
    cys_pcts_60 = []
    asn_pcts_60 = []
    gln_pcts_60 = []
    lengths_60 = []
    
    for i in idx_60:
        seq = sequences[i]
        lengths_60.append(len(seq))
        cys_pcts_60.append(seq.count('C') / len(seq) * 100)
        asn_pcts_60.append(seq.count('N') / len(seq) * 100)
        gln_pcts_60.append(seq.count('Q') / len(seq) * 100)
        
    lengths_60 = np.array(lengths_60)
    cys_pcts_60 = np.array(cys_pcts_60)
    asn_pcts_60 = np.array(asn_pcts_60)
    gln_pcts_60 = np.array(gln_pcts_60)
    
    outliers_60_idx = np.where((cys_pcts_60 > 4.0) | (asn_pcts_60 > 8.0) | (gln_pcts_60 > 6.0) | (lengths_60 > 1000))[0]
    print(f"Total outliers to clean at 60-80°C: {len(outliers_60_idx)} sequences ({len(outliers_60_idx)/len(idx_60)*100:.2f}%)")

if __name__ == "__main__":
    main()
