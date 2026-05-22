import torch
import json

def main():
    input_path = "/home/bibhu/Documents/temstampto/new_data/prepared_data_v3.pt"
    lookup_input_path = "/home/bibhu/Documents/temstampto/new_data/tm_ogt_lookup.json"
    lookup_output_path = "/home/bibhu/Documents/temstampto/new_data/tm_ogt_lookup.json"
    
    print(f"Loading UniProt mapping from {lookup_input_path}...")
    with open(lookup_input_path, "r") as f:
        uid_to_info = json.load(f)
        
    print(f"Loading dataset from {input_path}...")
    dataset = torch.load(input_path, map_location="cpu")
    
    # Define biologically verified OGTs for model organisms in Meltome/TemBERTure
    TAXID_TO_OGT = {
        "10090": 37.0,     # Mus musculus (Mouse)
        "6239": 20.0,      # Caenorhabditis elegans (Worm)
        "9606": 37.0,      # Homo sapiens (Human)
        "83333": 37.0,     # Escherichia coli K-12
        "559292": 30.0,    # Saccharomyces cerevisiae S288C (Yeast)
        "3702": 22.0,      # Arabidopsis thaliana (Plant)
        "507601": 75.0,    # Thermus thermophilus HB8 (Extreme thermophile)
        "300852": 80.0,    # Thermotoga maritima MSB8 (Hyperthermophile)
        "7227": 25.0,      # Drosophila melanogaster (Fly)
        "224308": 37.0,    # Bacillus subtilis str. 168
        "262724": 37.0,    # Streptococcus pneumoniae R6
        "698738": 60.0,    # Picrophilus torridus DSM 9790 (Acidophile/thermophile)
        "1122961": 50.0,   # Chaetomium thermophilum var. thermophilum
        "7955": 28.0,      # Danio rerio (Zebrafish)
        "7240": 25.0,      # Drosophila simulans
        "274": 60.0,       # Picrophilus torridus (Genus-level or general)
        "7245": 25.0,      # Drosophila yakuba
        "7238": 25.0,      # Drosophila sechellia
        "7220": 25.0,      # Drosophila erecta
        "10116": 37.0,     # Rattus norvegicus (Rat)
        "562": 37.0,       # Escherichia coli
        "4932": 30.0,      # Saccharomyces cerevisiae
    }
    
    # Update lookup with verified OGTs
    matched_count = 0
    predicted_needed = 0
    
    for uid, info in uid_to_info.items():
        taxid = info.get("taxid")
        if taxid in TAXID_TO_OGT:
            info["ogt"] = TAXID_TO_OGT[taxid]
            info["source"] = "known"
            matched_count += 1
        else:
            info["source"] = "needs_prediction"
            predicted_needed += 1
            
    print(f"Updated lookup mapping:")
    print(f"  Matched with verified OGT: {matched_count}")
    print(f"  Need prediction fallback:  {predicted_needed}")
    
    print(f"Saving updated mapping to {lookup_output_path}...")
    with open(lookup_output_path, "w") as f:
        json.dump(uid_to_info, f, indent=4)
    print("Done!")

if __name__ == "__main__":
    main()
