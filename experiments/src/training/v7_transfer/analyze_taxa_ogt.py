import torch
import json
import numpy as np

def main():
    data_path = "/home/bibhu/Documents/temstampto/data/embeddings/prepared_data_v4.pt"
    taxid_path = "/home/bibhu/Documents/temstampto/data/cleaner_data/taxid_lineages.json"
    
    print("Loading dataset...")
    dataset = torch.load(data_path, map_location='cpu')
    train_ogt = dataset["train_ogt"]
    tax_ids = train_ogt["tax_id"]
    labels = train_ogt["labels"] if "labels" in train_ogt else train_ogt["ogt_original"]
    
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    labels = np.array(labels)
    
    print("Loading taxonomy...")
    with open(taxid_path, "r") as f:
        taxid_db = json.load(f)
        
    # Analyze kingdom distribution
    superkingdom_counts = {}
    superkingdom_temps = {}
    
    for tid, temp in zip(tax_ids, labels):
        tid_str = str(tid)
        info = taxid_db.get(tid_str, {"superkingdom": "Unknown"})
        sk = info["superkingdom"]
        
        superkingdom_counts[sk] = superkingdom_counts.get(sk, 0) + 1
        if sk not in superkingdom_temps:
            superkingdom_temps[sk] = []
        superkingdom_temps[sk].append(temp)
        
    print("\n--- Kingdom / Superkingdom Distribution ---")
    for sk, count in superkingdom_counts.items():
        temps = np.array(superkingdom_temps[sk])
        print(f"Kingdom: {sk:<15} | Count: {count:<8} ({count/len(tax_ids)*100:.2f}%) | Mean Temp: {temps.mean():.2f}°C | Min: {temps.min():.2f}°C | Max: {temps.max():.2f}°C")
        
    # Analyze eukaryotic sequences with high OGT
    if "Eukaryota" in superkingdom_temps:
        euk_temps = np.array(superkingdom_temps["Eukaryota"])
        high_euk_temps = euk_temps[euk_temps > 45.0]
        print(f"\nEukaryotic sequences with OGT > 45°C: {len(high_euk_temps)} out of {len(euk_temps)} ({len(high_euk_temps)/len(euk_temps)*100:.2f}%)")
        high_euk_temps_60 = euk_temps[euk_temps > 60.0]
        print(f"Eukaryotic sequences with OGT > 60°C: {len(high_euk_temps_60)} ({len(high_euk_temps_60)/len(euk_temps)*100:.2f}%)")
        
        # Let's list some eukaryotic organisms with high OGT to see if they make sense
        high_euk_orgs = {}
        for tid, temp in zip(tax_ids, labels):
            tid_str = str(tid)
            info = taxid_db.get(tid_str, {"superkingdom": "Unknown"})
            if info["superkingdom"] == "Eukaryota" and temp > 45.0:
                name = info["name"]
                if name not in high_euk_orgs:
                    high_euk_orgs[name] = (temp, 0)
                high_euk_orgs[name] = (temp, high_euk_orgs[name][1] + 1)
                
        high_euk_orgs = sorted(list(high_euk_orgs.items()), key=lambda x: x[1][0], reverse=True)
        print("\nHigh temperature eukaryotic organisms in dataset:")
        for name, (temp, count) in high_euk_orgs[:15]:
            print(f"  Organism: {name:<40} | Temp: {temp:.2f}°C | Count: {count}")
            
if __name__ == "__main__":
    main()
