import pandas as pd
import argparse
import os

def load_ogt_reference(ref_path):
    # This might be tricky depending on the exact format of ogt_per_organism.tsv
    # Assuming it has columns like 'organism' and 'ogt' or similar
    # If the user doesn't have an exact file like this, we'll need to parse the fasta headers
    print(f"Loading OGT reference from {ref_path}...")
    # This is a placeholder logic based on typical TSV mapping
    try:
        df = pd.read_csv(ref_path, sep='\t')
        # Try to find species/organism and OGT columns
        species_col = next((col for col in df.columns if 'species' in col.lower() or 'organism' in col.lower()), None)
        ogt_col = next((col for col in df.columns if 'ogt' in col.lower() or 'temp' in col.lower()), None)
        
        if species_col and ogt_col:
            # Create mapping
            mapping = dict(zip(df[species_col].str.lower(), df[ogt_col]))
            return mapping
        else:
            print("Could not identify organism/ogt columns automatically.")
            return {}
    except Exception as e:
        print(f"Error loading reference: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--meltome-csv', type=str, default='new_data/meltome_sequences.csv')
    parser.add_argument('--prothermdb-csv', type=str, default='new_data/prothermdb_validation.csv')
    parser.add_argument('--ogt-reference', type=str, default='../dataset/ogt_per_organism.tsv')
    parser.add_argument('--output-dir', type=str, default='new_data/')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Try to load OGT mapping
    # Note: If the actual mapping is not available, we'll just create dummy logic for now
    ogt_mapping = load_ogt_reference(args.ogt_reference)
    
    if not ogt_mapping:
        print("Warning: OGT mapping is empty. Will fill with NaNs.")
        
    for file_path in [args.meltome_csv, args.prothermdb_csv]:
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist, skipping.")
            continue
            
        print(f"Processing {file_path}...")
        df = pd.read_csv(file_path)
        
        # Identify organism column
        org_col = next((col for col in df.columns if 'organism' in col.lower() or 'species' in col.lower()), None)
        
        if org_col:
            df['OGT'] = df[org_col].astype(str).str.lower().map(ogt_mapping)
            print(f"Mapped {df['OGT'].notna().sum()} out of {len(df)} records.")
        else:
            print("No organism column found.")
            df['OGT'] = float('nan')
            
        out_name = os.path.basename(file_path).replace('.csv', '_with_ogt.csv')
        out_path = os.path.join(args.output_dir, out_name)
        df.to_csv(out_path, index=False)
        print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
