"""
Rebuild OGT lookup from scratch.
Reads the existing build_ogt_lookup output (with taxids) and applies 
verified organism growth temperatures for the ~20 model organisms that 
cover 99%+ of the Tm dataset.

Saves to cleaned_data/ only.
"""
import json
import torch
import csv
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

# Biologically verified OGTs for the model organisms in Meltome/TemBERTure
TAXID_TO_OGT = {
    "10090": 37.0,     # Mus musculus (Mouse)
    "6239": 20.0,      # Caenorhabditis elegans (Worm)
    "9606": 37.0,      # Homo sapiens (Human)
    "83333": 37.0,     # Escherichia coli K-12 MG1655
    "83334": 37.0,     # Escherichia coli O157:H7
    "559292": 30.0,    # Saccharomyces cerevisiae S288C (Yeast)
    "3702": 22.0,      # Arabidopsis thaliana (Plant)
    "507601": 75.0,    # Thermus thermophilus HB8 (Extreme thermophile)
    "300852": 80.0,    # Thermotoga maritima MSB8 (Hyperthermophile)
    "7227": 25.0,      # Drosophila melanogaster (Fly)
    "224308": 37.0,    # Bacillus subtilis str. 168
    "262724": 37.0,    # Streptococcus pneumoniae R6
    "698738": 60.0,    # Picrophilus torridus DSM 9790
    "1122961": 50.0,   # Chaetomium thermophilum
    "7955": 28.0,      # Danio rerio (Zebrafish)
    "7240": 25.0,      # Drosophila simulans
    "274": 60.0,       # Picrophilus torridus
    "7245": 25.0,      # Drosophila yakuba
    "7238": 25.0,      # Drosophila sechellia
    "7220": 25.0,      # Drosophila erecta
    "10116": 37.0,     # Rattus norvegicus (Rat)
    "562": 37.0,       # Escherichia coli (general)
    "4932": 30.0,      # Saccharomyces cerevisiae (general)
}

def query_uniprot_batch(ids_batch):
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
            res = requests.get(url, params=params, timeout=30)
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
                time.sleep(2 * (attempt + 1))
            else:
                time.sleep(1)
        except Exception:
            time.sleep(1)
    return {}

def main():
    base = "/home/bibhu/Documents/temstampto"
    input_path = os.path.join(base, "cleaned_data/prepared_data_v3.pt")
    output_path = os.path.join(base, "cleaned_data/tm_ogt_lookup.json")
    
    print("Loading cleaned dataset...")
    dataset = torch.load(input_path, map_location="cpu")
    
    # 1. Build TaxID -> OGT from train_ogt (prokaryotic database)
    print("Building TaxID -> OGT from OGT training set...")
    taxid_to_ogt = dict(TAXID_TO_OGT)  # Start with verified eukaryotic OGTs
    
    for full_id in dataset["train_ogt"]["ids"]:
        parts = full_id.split("|")
        if len(parts) >= 3:
            taxid_to_ogt[parts[0]] = float(parts[-1])
    print(f"  Total organisms with known OGT: {len(taxid_to_ogt)}")
    
    # 2. Collect all unique UniProt IDs from Tm datasets
    print("Collecting unique UniProt IDs...")
    uids = set()
    for split in ["train_tm", "val_tm", "test_tm"]:
        for full_id in dataset[split]["ids"]:
            uids.add(full_id.split("|")[0])
    uids = sorted(list(uids))
    print(f"  Total unique UniProt IDs: {len(uids)}")
    
    # 3. Query UniProt for TaxIDs (parallel, batch size 80)
    batch_size = 80
    batches = [uids[i:i+batch_size] for i in range(0, len(uids), batch_size)]
    uid_to_taxid = {}
    
    print(f"Querying UniProt for TaxIDs ({len(batches)} batches, 10 workers)...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(query_uniprot_batch, b): b for b in batches}
        for future in tqdm(as_completed(futures), total=len(batches)):
            try:
                uid_to_taxid.update(future.result())
            except Exception as e:
                print(f"Batch error: {e}")
    
    print(f"  UniProt returned TaxIDs for: {len(uid_to_taxid)}/{len(uids)}")
    
    # 4. Map UniProt ID -> OGT via TaxID
    uid_to_ogt = {}
    matched = 0
    taxid_no_ogt = 0
    no_taxid = 0
    
    for uid in uids:
        if uid in uid_to_taxid:
            taxid = uid_to_taxid[uid]
            if taxid in taxid_to_ogt:
                uid_to_ogt[uid] = {
                    "ogt": taxid_to_ogt[taxid],
                    "taxid": taxid,
                    "source": "known"
                }
                matched += 1
            else:
                uid_to_ogt[uid] = {
                    "taxid": taxid,
                    "source": "needs_prediction"
                }
                taxid_no_ogt += 1
        else:
            uid_to_ogt[uid] = {"source": "needs_prediction"}
            no_taxid += 1
    
    # 5. Also map FireProt holdout sequences
    print("\nMapping FireProt holdout sequences...")
    holdout_path = os.path.join(base, "experiments/data_processing/fireprot_holdout_prott5.pt")
    sql_path = os.path.join(base, "new_data/fireprotdb_dump_2025_09_22/01_fireprotdb_2025-09-20.sql")
    csv_path = os.path.join(base, "new_data/fireprotdb_dump_2025_09_22/fireprotdb_csv_whole/fireprotdb_20251015-164116.csv")
    
    fp = torch.load(holdout_path, map_location="cpu")
    # Parse SQL sequences
    sequences = {}
    in_copy = False
    with open(sql_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("COPY public.sequence "):
                in_copy = True
                continue
            if in_copy:
                if line.strip() == "\\.":
                    break
                parts = line.split("\t")
                if len(parts) >= 2:
                    sequences[parts[0].strip()] = parts[1].strip().upper()
    seq_to_id = {seq: sid for sid, seq in sequences.items()}
    
    sid_to_uniprot = {}
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) > 38:
                sid_to_uniprot[row[1].strip()] = row[38].strip()
    
    fp_uids = []
    for seq in fp["sequences"]:
        sid = seq_to_id.get(seq)
        fp_uids.append(sid_to_uniprot.get(sid, "") if sid else "")
    
    # Query UniProt for FireProt TaxIDs
    fp_unique = sorted(set(u for u in fp_uids if u))
    fp_batches = [fp_unique[i:i+80] for i in range(0, len(fp_unique), 80)]
    fp_taxids = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(query_uniprot_batch, b) for b in fp_batches]
        for fut in futures:
            fp_taxids.update(fut.result())
    
    fp_matched = 0
    for uid in fp_unique:
        if uid in fp_taxids:
            taxid = fp_taxids[uid]
            if taxid in taxid_to_ogt:
                uid_to_ogt[uid] = {"ogt": taxid_to_ogt[taxid], "taxid": taxid, "source": "known"}
                fp_matched += 1
            else:
                uid_to_ogt[uid] = {"taxid": taxid, "source": "needs_prediction"}
        else:
            uid_to_ogt[uid] = {"source": "needs_prediction"}
    
    print(f"  FireProt: {fp_matched}/{len(fp_unique)} matched to OGT")
    
    # Summary
    total = len(uid_to_ogt)
    known_total = sum(1 for v in uid_to_ogt.values() if v.get("source") == "known")
    needs_total = total - known_total
    
    print(f"\n{'='*60}")
    print(f"OGT LOOKUP RESULTS")
    print(f"{'='*60}")
    print(f"  Total proteins:        {total:,}")
    print(f"  Matched with OGT:      {known_total:,} ({known_total/total*100:.1f}%)")
    print(f"  Needs prediction:      {needs_total:,} ({needs_total/total*100:.1f}%)")
    
    print(f"\nSaving to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(uid_to_ogt, f, indent=2)
    print("Done!")

if __name__ == "__main__":
    main()
