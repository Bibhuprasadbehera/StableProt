import os
import pandas as pd
import matplotlib.pyplot as plt

def extract_fasta(df, output_path):
    # Get unique sequences
    unique_proteins = df[['Protein_ID', 'Sequence', 'Tm']].drop_duplicates(subset=['Protein_ID'])
    
    with open(output_path, 'w') as f:
        for _, row in unique_proteins.iterrows():
            f.write(f">{row['Protein_ID']}|{row['Tm']}\n")
            f.write(f"{row['Sequence']}\n")
            
    return len(unique_proteins)

def main():
    repo_dir = "TemBERTure_repo/data"
    files = {
        "Train": "TemBERTureTrain_reg.txt",
        "Val": "TemBERTureVal_reg.txt",
        "Test": "TemBERTureTest_reg.txt"
    }
    
    os.makedirs("new_data", exist_ok=True)
    all_dfs = []
    
    print("Analyzing TemBERTureDB Regression Data...")
    
    for split, filename in files.items():
        filepath = os.path.join(repo_dir, filename)
        if not os.path.exists(filepath):
            print(f"Error: {filepath} not found.")
            continue
            
        df = pd.read_csv(filepath)
        df['Split'] = split
        all_dfs.append(df)
        
        print(f"\n--- {split} Split ---")
        print(f"Total rows: {len(df)}")
        print(f"Unique proteins: {df['Protein_ID'].nunique()}")
        print(f"Tm range: {df['Tm'].min():.1f} to {df['Tm'].max():.1f}")
        
    if not all_dfs:
        return
        
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Extract FASTA for the combined regression sequences (train + val)
    # We'll save test separately as requested by Phase 1.6/5.4
    train_val_df = combined[combined['Split'].isin(['Train', 'Val'])]
    train_val_fasta = "new_data/tembert_reg_sequences.fasta"
    count_train_val = extract_fasta(train_val_df, train_val_fasta)
    print(f"\nExtracted {count_train_val} unique sequences to {train_val_fasta}")
    
    test_df = combined[combined['Split'] == 'Test']
    test_fasta = "new_data/tembert_test_sequences.fasta"
    count_test = extract_fasta(test_df, test_fasta)
    print(f"Extracted {count_test} unique sequences to {test_fasta}")
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(combined['Tm'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title("Tm Distribution in TemBERTureDB Regression Data")
    plt.xlabel("Melting Temperature (Tm) °C")
    plt.ylabel("Count")
    plt.grid(axis='y', alpha=0.3)
    
    os.makedirs("experiments/results", exist_ok=True)
    plt.savefig("experiments/results/tembert_tm_dist.png")
    plt.close()
    print("Saved histogram to experiments/results/tembert_tm_dist.png")

if __name__ == "__main__":
    main()
