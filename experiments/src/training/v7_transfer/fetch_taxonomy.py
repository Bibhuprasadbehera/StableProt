import torch
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import json
import os
import time

def fetch_batch(taxids):
    taxids_str = ",".join(map(str, taxids))
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=taxonomy&id={taxids_str}&retmode=xml"
    
    # Try fetching with retries
    for attempt in range(5):
        try:
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                return response.read()
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}. Retrying in 2 seconds...")
            time.sleep(2)
    return None

def parse_xml_to_dict(xml_data):
    results = {}
    if not xml_data:
        return results
        
    try:
        root = ET.fromstring(xml_data)
        for taxon in root.findall('Taxon'):
            taxid_node = taxon.find('TaxId')
            name_node = taxon.find('ScientificName')
            lineage_node = taxon.find('Lineage')
            
            if taxid_node is not None:
                taxid = taxid_node.text
                name = name_node.text if name_node is not None else ""
                lineage = lineage_node.text if lineage_node is not None else ""
                
                # Determine superkingdom/kingdom
                superkingdom = "Unknown"
                if "Eukaryota" in lineage:
                    superkingdom = "Eukaryota"
                elif "Bacteria" in lineage:
                    superkingdom = "Bacteria"
                elif "Archaea" in lineage:
                    superkingdom = "Archaea"
                elif "Viruses" in lineage:
                    superkingdom = "Viruses"
                elif "cellular organisms" in lineage:
                    # check if lineage contains other info
                    pass
                    
                results[taxid] = {
                    "name": name,
                    "superkingdom": superkingdom,
                    "lineage": lineage
                }
    except Exception as e:
        print(f"Error parsing XML: {e}")
    return results

def main():
    data_path = "/home/bibhu/Documents/temstampto/data/embeddings/prepared_data_v4.pt"
    out_path = "/home/bibhu/Documents/temstampto/data/cleaner_data/taxid_lineages.json"
    
    print("Loading dataset to find unique taxids...")
    dataset = torch.load(data_path, map_location='cpu')
    tax_ids = dataset["train_ogt"]["tax_id"]
    unique_taxids = sorted(list(set(map(int, tax_ids))))
    
    print(f"Found {len(unique_taxids)} unique TaxIDs.")
    
    # Load existing lookups if any
    taxid_db = {}
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            taxid_db = json.load(f)
        print(f"Loaded {len(taxid_db)} taxids from existing cache.")
        
    # Find which ones are missing
    missing_taxids = [tid for tid in unique_taxids if str(tid) not in taxid_db]
    print(f"Need to fetch: {len(missing_taxids)} taxids.")
    
    if not missing_taxids:
        print("All taxids already cached!")
        return
        
    # Query in batches of 200
    batch_size = 200
    for i in range(0, len(missing_taxids), batch_size):
        batch = missing_taxids[i:i+batch_size]
        print(f"Fetching batch {i//batch_size + 1}/{(len(missing_taxids)+batch_size-1)//batch_size} (size {len(batch)})...")
        
        xml_data = fetch_batch(batch)
        if xml_data:
            parsed = parse_xml_to_dict(xml_data)
            taxid_db.update(parsed)
            print(f"  Successfully fetched and parsed {len(parsed)} taxids. Cache size: {len(taxid_db)}")
            
            # Save intermediate progress
            with open(out_path, "w") as f:
                json.dump(taxid_db, f, indent=2)
        else:
            print(f"  Failed to fetch batch {batch}")
            
        time.sleep(0.5) # rate limit compliance
        
    print(f"Done! Final cache size: {len(taxid_db)}")

if __name__ == "__main__":
    main()
