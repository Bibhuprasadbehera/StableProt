import os
import pandas as pd
import matplotlib.pyplot as plt
import requests
import time
from tqdm import tqdm

def fetch_fasta_from_uniprot(uniprot_ids, batch_size=500):
    """
    Fetches FASTA sequences from UniProt REST API in batches.
    """
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
                
                # Parse FASTA
                fasta_text = response.text
                entries = fasta_text.split('>')[1:]
                for entry in entries:
                    lines = entry.strip().split('\n')
                    if not lines: continue
                    header = lines[0]
                    # Format: tr|A0A024RBG1|... or sp|P12345|...
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
    csv_path = "new_data/meltome-atlas/cross-species.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Extract UniProt ID (e.g., 'C0H3Q1_ytzI' -> 'C0H3Q1')
    # Some might be just the ID
    df['UniProt_ID'] = df['Protein_ID'].astype(str).apply(lambda x: x.split('_')[0] if '_' in x else x)
    
    # Filter out NA meltPoint
    df = df.dropna(subset=['meltPoint'])
    
    # Aggregate median meltPoint per protein
    agg_df = df.groupby('UniProt_ID')['meltPoint'].median().reset_index()
    agg_df.rename(columns={'meltPoint': 'Tm'}, inplace=True)
    
    # Optional: extract species if available in run_name
    # run_name is like "Bacillus subtilis_168_lysate_R1"
    df['organism'] = df['run_name'].astype(str).apply(lambda x: x.split('_lysate')[0] if '_lysate' in x else x.split('_')[0])
    org_map = df.groupby('UniProt_ID')['organism'].first().reset_index()
    agg_df = pd.merge(agg_df, org_map, on='UniProt_ID', how='left')
    
    print(f"Found {len(agg_df)} unique proteins with Tm values.")
    print(f"Tm range: {agg_df['Tm'].min():.1f} to {agg_df['Tm'].max():.1f}")
    
    # Save the aggregated CSV
    out_csv = "new_data/meltome_sequences.csv"
    agg_df.to_csv(out_csv, index=False)
    print(f"Saved aggregated data to {out_csv}")
    
    # Fetch FASTA
    fastas = fetch_fasta_from_uniprot(agg_df['UniProt_ID'].tolist())
    
    print(f"Successfully fetched {len(fastas)} out of {len(agg_df)} sequences.")
    
    out_fasta = "new_data/meltome_sequences.fasta"
    with open(out_fasta, 'w') as f:
        for _, row in agg_df.iterrows():
            uid = row['UniProt_ID']
            tm = row['Tm']
            if uid in fastas:
                f.write(f">{uid}|{tm}\n")
                f.write(f"{fastas[uid]}\n")
                
    print(f"Saved sequences to {out_fasta}")
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(agg_df['Tm'], bins=50, alpha=0.7, color='green', edgecolor='black')
    plt.title("Tm Distribution in Meltome Atlas Data")
    plt.xlabel("Melting Temperature (Tm) °C")
    plt.ylabel("Count")
    plt.grid(axis='y', alpha=0.3)
    
    os.makedirs("experiments/results", exist_ok=True)
    plt.savefig("experiments/results/meltome_tm_dist.png")
    plt.close()
    print("Saved histogram to experiments/results/meltome_tm_dist.png")

if __name__ == "__main__":
    main()
