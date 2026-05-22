import torch
import requests
import json
import os
import time
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

def query_uniprot_batch(ids_batch):
    # Construct search query with ORs
    query = " OR ".join([f"accession:{uid}" for uid in ids_batch])
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": query,
        "fields": "accession,organism_id",
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
                    taxon_id = record.get("organism", {}).get("taxonId")
                    if acc and taxon_id:
                        mapping[acc] = str(taxon_id)
                return mapping
            elif res.status_code == 429:
                # Rate limit, sleep and retry
                time.sleep(2 * (attempt + 1))
            else:
                # Print debug or just retry
                time.sleep(1)
        except Exception:
            time.sleep(1)
            
    return {}

def main():
    input_path = "/home/bibhu/Documents/temstampto/new_data/prepared_data_v3.pt"
    lookup_output_path = "/home/bibhu/Documents/temstampto/new_data/tm_ogt_lookup.json"
    
    print(f"Loading dataset from {input_path}...")
    dataset = torch.load(input_path, map_location="cpu")
    
    # 1. Build Taxon ID -> OGT lookup from train_ogt
    print("Building Taxon ID -> OGT mapping from OGT dataset...")
    taxid_to_ogt = {}
    ogt_ids = dataset["train_ogt"]["ids"]
    ogt_labels = dataset["train_ogt"]["labels"].numpy()
    
    for i, full_id in enumerate(ogt_ids):
        parts = full_id.split("|")
        if len(parts) >= 3:
            taxid = parts[0]
            ogt = float(parts[-1])
            taxid_to_ogt[taxid] = ogt
            
    print(f"Loaded OGT for {len(taxid_to_ogt)} unique taxon IDs.")
    
    # 2. Extract all unique UniProt IDs from Tm datasets
    print("Collecting unique UniProt IDs from Tm datasets...")
    uids = set()
    for split in ["train_tm", "val_tm", "test_tm"]:
        for full_id in dataset[split]["ids"]:
            uids.add(full_id.split("|")[0])
    
    uids = sorted(list(uids))
    print(f"Total unique UniProt IDs to map: {len(uids)}")
    
    # 3. Query UniProt search API in parallel batches of 80
    batch_size = 80
    batches = [uids[i:i+batch_size] for i in range(0, len(uids), batch_size)]
    uid_to_taxid = {}
    
    print(f"Querying UniProt in parallel batches (n={len(batches)} batches, 10 workers)...")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks
        future_to_batch = {executor.submit(query_uniprot_batch, batch): batch for batch in batches}
        
        # Monitor progress with tqdm
        for future in tqdm(as_completed(future_to_batch), total=len(batches)):
            try:
                mapping = future.result()
                uid_to_taxid.update(mapping)
            except Exception as e:
                print(f"Batch failed: {e}")
                
    print(f"Mapped {len(uid_to_taxid)} / {len(uids)} UniProt IDs to Taxon ID.")
    
    # 4. Map UniProt ID -> OGT
    uid_to_ogt = {}
    matched_count = 0
    missing_taxid_count = 0
    missing_ogt_count = 0
    
    for uid in uids:
        if uid in uid_to_taxid:
            taxid = uid_to_taxid[uid]
            if taxid in taxid_to_ogt:
                uid_to_ogt[uid] = {
                    "ogt": taxid_to_ogt[taxid],
                    "taxid": taxid,
                    "source": "known"
                }
                matched_count += 1
            else:
                uid_to_ogt[uid] = {
                    "taxid": taxid,
                    "source": "missing_ogt_database"
                }
                missing_ogt_count += 1
        else:
            uid_to_ogt[uid] = {
                "source": "missing_uniprot_taxid"
            }
            missing_taxid_count += 1
            
    print(f"Mapping Results:")
    print(f"  Matched with OGT: {matched_count} ({matched_count/len(uids)*100:.1f}%)")
    print(f"  TaxID found but no OGT in database: {missing_ogt_count}")
    print(f"  UniProt TaxID not found: {missing_taxid_count}")
    
    print(f"Saving lookup mapping to {lookup_output_path}...")
    with open(lookup_output_path, "w") as f:
        json.dump(uid_to_ogt, f, indent=4)
    print("OGT lookup file built successfully!")

if __name__ == "__main__":
    main()
