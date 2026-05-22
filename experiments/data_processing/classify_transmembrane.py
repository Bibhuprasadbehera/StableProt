import torch
import requests
import json
import os
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def query_uniprot_transmem_batch(ids_batch):
    query = " OR ".join([f"accession:{uid}" for uid in ids_batch])
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": query,
        "fields": "accession,ft_transmem",
        "format": "json",
        "size": len(ids_batch) * 2
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            res = requests.get(url, params=params, timeout=20)
            if res.status_code == 200:
                data = res.json()
                mapping = {}
                for record in data.get("results", []):
                    acc = record.get("primaryAccession")
                    # Check if transmembrane features are present
                    features = record.get("features", [])
                    has_tm = 0
                    for f in features:
                        if f.get("type") == "Transmembrane":
                            has_tm = 1
                            break
                    if acc:
                        mapping[acc] = has_tm
                return mapping
            elif res.status_code == 429:
                time.sleep(2 * (attempt + 1))
            else:
                time.sleep(1)
        except Exception:
            time.sleep(1)
            
    return {}

def main():
    input_path = "/home/bibhu/Documents/temstampto/new_data/prepared_data_v3.pt"
    output_path = "/home/bibhu/Documents/temstampto/new_data/tm_transmembrane.json"
    
    print(f"Loading dataset from {input_path}...")
    dataset = torch.load(input_path, map_location="cpu")
    
    # Extract all unique UniProt IDs from Tm datasets
    print("Collecting unique UniProt IDs...")
    uids = set()
    for split in ["train_tm", "val_tm", "test_tm"]:
        for full_id in dataset[split]["ids"]:
            uids.add(full_id.split("|")[0])
    
    uids = sorted(list(uids))
    print(f"Total unique UniProt IDs: {len(uids)}")
    
    # Query UniProt search API in parallel batches of 80
    batch_size = 80
    batches = [uids[i:i+batch_size] for i in range(0, len(uids), batch_size)]
    uid_to_tm = {}
    
    print(f"Querying UniProt for transmembrane features (n={len(batches)} batches, 10 workers)...")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_batch = {executor.submit(query_uniprot_transmem_batch, batch): batch for batch in batches}
        
        for future in tqdm(as_completed(future_to_batch), total=len(batches)):
            try:
                mapping = future.result()
                uid_to_tm.update(mapping)
            except Exception as e:
                print(f"Batch failed: {e}")
                
    # Fill in default 0 (soluble) for any accession that failed to map
    filled_default = 0
    for uid in uids:
        if uid not in uid_to_tm:
            uid_to_tm[uid] = 0
            filled_default += 1
            
    tm_count = sum(uid_to_tm.values())
    soluble_count = len(uid_to_tm) - tm_count
    
    print(f"Transmembrane Feature mapping results:")
    print(f"  Transmembrane proteins (1): {tm_count} ({tm_count/len(uids)*100:.1f}%)")
    print(f"  Soluble proteins (0):       {soluble_count} ({soluble_count/len(uids)*100:.1f}%)")
    print(f"  Filled with default 0:      {filled_default}")
    
    print(f"Saving mapping to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(uid_to_tm, f, indent=4)
    print("Transmembrane mapping file built successfully!")

if __name__ == "__main__":
    main()
