import os
import pandas as pd
import requests
import time
from tqdm import tqdm

def fetch_fasta_from_uniprot(uniprot_ids, batch_size=500):
    url = 'https://rest.uniprot.org/uniprotkb/accessions'
    all_fastas = {}
    
    batches = [uniprot_ids[i:i + batch_size] for i in range(0, len(uniprot_ids), batch_size)]
    
    print(f"Fetching {len(uniprot_ids)} sequences from UniProt in {len(batches)} batches...")
    
    for batch in tqdm(batches):
        params = {
            'accessions': ','.join(batch),
            'format': 'fasta'
        }
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                fasta_text = response.text
                entries = fasta_text.split('>')[1:]
                for entry in entries:
                    lines = entry.strip().split('\n')
                    if not lines: continue
                    header = lines[0]
                    parts = header.split('|')
                    if len(parts) >= 2:
                        uid = parts[1]
                        seq = ''.join(lines[1:])
                        all_fastas[uid] = seq
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"\nFailed to fetch batch after {max_retries} attempts: {e}")
                time.sleep(2)
                
    return all_fastas

def main():
    protherm_dir = "new_data/prothermadb"
    if not os.path.exists(protherm_dir):
        print(f"Error: {protherm_dir} not found.")
        return
        
    all_dfs = []
    
    for file in os.listdir(protherm_dir):
        if file.endswith(".xlsx"):
            path = os.path.join(protherm_dir, file)
            print(f"Reading {file}...")
            df = pd.read_excel(path)
            
            # Print columns to help debug if they change
            # print(f"Columns in {file}: {df.columns.tolist()}")
            
            # Standardize column names (make uppercase to avoid case issues)
            df.columns = [str(c).upper().strip() for c in df.columns]
            
            all_dfs.append(df)
            
    if not all_dfs:
        print("No xlsx files found.")
        return
        
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"Total rows in ProThermDB: {len(combined)}")
    
    # Check for MUTATION column
    mut_col = next((c for c in combined.columns if 'MUTATION' in c), None)
    tm_col = next((c for c in combined.columns if 'TM' in c), None)
    uid_col = next((c for c in combined.columns if 'UNIPROT' in c), None)
    org_col = next((c for c in combined.columns if 'ORGANISM' in c or 'SOURCE' in c), None)
    
    if not mut_col or not tm_col or not uid_col:
        print(f"Error: Missing essential columns. Found: {combined.columns.tolist()}")
        return
        
    # Filter for wild type
    wt_df = combined[combined[mut_col].astype(str).str.lower() == 'wild']
    print(f"Wild-type rows: {len(wt_df)}")
    
    # Filter for non-null Tm
    wt_df = wt_df.dropna(subset=[tm_col, uid_col])
    
    # Convert Tm to numeric, coercing errors
    wt_df[tm_col] = pd.to_numeric(wt_df[tm_col], errors='coerce')
    wt_df = wt_df.dropna(subset=[tm_col])
    
    print(f"Valid wild-type rows with Tm: {len(wt_df)}")
    
    # Deduplicate by UniProt ID (take median Tm)
    agg_df = wt_df.groupby(uid_col)[tm_col].median().reset_index()
    agg_df.rename(columns={uid_col: 'UniProt_ID', tm_col: 'Tm'}, inplace=True)
    
    if org_col:
        org_map = wt_df.groupby(uid_col)[org_col].first().reset_index()
        org_map.rename(columns={uid_col: 'UniProt_ID', org_col: 'Organism'}, inplace=True)
        agg_df = pd.merge(agg_df, org_map, on='UniProt_ID', how='left')
    
    print(f"Unique proteins after deduplication: {len(agg_df)}")
    
    # Save CSV
    out_csv = "new_data/prothermdb_validation.csv"
    agg_df.to_csv(out_csv, index=False)
    print(f"Saved to {out_csv}")
    
    # Fetch FASTA
    fastas = fetch_fasta_from_uniprot(agg_df['UniProt_ID'].tolist())
    
    print(f"Successfully fetched {len(fastas)} out of {len(agg_df)} sequences.")
    
    out_fasta = "new_data/prothermdb_validation.fasta"
    with open(out_fasta, 'w') as f:
        for _, row in agg_df.iterrows():
            uid = row['UniProt_ID']
            tm = row['Tm']
            if uid in fastas:
                f.write(f">{uid}|{tm}\n")
                f.write(f"{fastas[uid]}\n")
                
    print(f"Saved sequences to {out_fasta}")

if __name__ == "__main__":
    main()
