import torch
import numpy as np

def main():
    data_path = "/home/bibhu/Documents/temstampto/data/embeddings/prepared_data_v4.pt"
    print(f"Loading dataset from {data_path}...")
    dataset = torch.load(data_path, map_location='cpu')
    
    train_ogt = dataset["train_ogt"]
    tax_ids = train_ogt["tax_id"]
    labels = train_ogt["labels"] if "labels" in train_ogt else train_ogt["ogt_original"]
    
    # If labels is a tensor, convert to numpy
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    labels = np.array(labels)
    
    unique_taxids = list(set(tax_ids))
    print(f"Total sequences: {len(tax_ids)}")
    print(f"Unique TaxIDs: {len(unique_taxids)}")
    
    # Check some temperature stats
    print(f"Min temperature: {labels.min():.2f}°C")
    print(f"Max temperature: {labels.max():.2f}°C")
    print(f"Mean temperature: {labels.mean():.2f}°C")
    
    # Count how many have temperature > 60
    high_temp_count = np.sum(labels > 60.0)
    print(f"Sequences with OGT > 60°C: {high_temp_count} ({high_temp_count/len(labels)*100:.2f}%)")
    
    # Group by taxid and find their temperatures
    taxid_to_temps = {}
    for tid, temp in zip(tax_ids, labels):
        if tid not in taxid_to_temps:
            taxid_to_temps[tid] = []
        taxid_to_temps[tid].append(temp)
        
    taxid_stats = []
    for tid, temps in taxid_to_temps.items():
        temps = np.array(temps)
        taxid_stats.append((tid, temps[0], len(temps))) # since OGT is organism-level, all temps for a taxid should be identical
        
    # Sort by temperature desc
    taxid_stats.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 highest temperature TaxIDs:")
    for tid, temp, count in taxid_stats[:10]:
        print(f"TaxID: {tid:<10} | Temp: {temp:.2f}°C | Count: {count}")
        
if __name__ == "__main__":
    main()
