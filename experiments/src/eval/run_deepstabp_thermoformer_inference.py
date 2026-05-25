import os
import sys
import csv
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.append(PROJECT_ROOT)

# Import ThermoFormer
sys.path.append(os.path.join(PROJECT_ROOT, "benchmark_models/ThermoFormer"))
from model.modeling_thermoformer import ThermoFormer
from model.tokenization_thermoformer import ThermoFormerTokenizer

def map_fireprot_to_uniprot_ids(fp_seqs, base_dir):
    sql_path = "/home/bibhu/Documents/temstampto/data/training_data/raw/fireprotdb_dump_2025_09_22/01_fireprotdb_2025-09-20.sql"
    csv_path = "/home/bibhu/Documents/temstampto/data/training_data/raw/fireprotdb_dump_2025_09_22/fireprotdb_csv_whole/fireprotdb_20251015-164116.csv"
    
    # Read sequence dictionary from SQL dump
    sequences = {}
    in_copy_block = False
    with open(sql_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("COPY public.sequence "):
                in_copy_block = True
                continue
            if in_copy_block:
                if line.strip() == "\\.":
                    break
                parts = line.split("\t")
                if len(parts) >= 2:
                    sequences[parts[0].strip()] = parts[1].strip().upper()
                    
    seq_to_id = {seq: seq_id for seq_id, seq in sequences.items()}
    
    # Read mapping from CSV
    seq_id_to_uniprot = {}
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) > 38:
                seq_id_to_uniprot[row[1].strip()] = row[38].strip()
                
    uids = []
    for seq in fp_seqs:
        seq_id = seq_to_id.get(seq)
        if seq_id and seq_id in seq_id_to_uniprot:
            uids.append(seq_id_to_uniprot[seq_id])
        else:
            uids.append(None)
    return uids

# -------------------------------------------------------------
# DeepSTABp Model Definition (without PyTorch Lightning dependency)
# -------------------------------------------------------------
class deepSTAPpMLP(nn.Module):
    def __init__(self, dropout=0.1, learning_rate=0.01, batch_size=25):
        super().__init__()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dropout = dropout
        self.zero_layer = nn.Linear(1064, 4098)
        self.zero_dropout = nn.Dropout1d(dropout)
        self.first_layer = nn.Linear(4098, 512)
        self.first_dropout = nn.Dropout1d(dropout)
        self.second_layer = nn.Linear(512, 256)
        self.second_dropout = nn.Dropout1d(dropout)
        self.third_layer = nn.Linear(256, 128)
        self.third_dropout = nn.Dropout1d(dropout)
        self.seventh_layer = nn.Linear(128, 1)
        self.species_layer_one = nn.Linear(1, 20)
        self.species_layer_two = nn.Linear(20, 20)
        self.species_dropout = nn.Dropout1d(dropout)
        self.batch_norm0 = nn.LayerNorm(4098)
        self.batch_norm1 = nn.LayerNorm(512)
        self.batch_norm2 = nn.LayerNorm(256)
        self.batch_norm3 = nn.LayerNorm(128)
        self.lysate = nn.Linear(1, 20)
        self.lysate2 = nn.Linear(20, 10)
        self.lysate_dropout = nn.Dropout1d(dropout)
        self.cell = nn.Linear(1, 20)
        self.cell2 = nn.Linear(20, 10)
        self.cell_dropout = nn.Dropout1d(dropout)

    def forward(self, x, species_feature, lysate, cell):
        x = x.float()
        species_feature = species_feature.float().reshape(-1, 1)
        lysate = lysate.float().reshape(-1, 1)
        cell = cell.float().reshape(-1, 1)
        lysate = self.lysate_dropout(F.selu(self.lysate(lysate)))
        lysate = self.lysate_dropout(F.selu(self.lysate2(lysate)))
        cell = self.cell_dropout(F.selu(self.cell(cell)))
        cell = self.cell_dropout(F.selu(self.cell2(cell)))
        species_feature = self.species_dropout(F.selu(self.species_layer_one(species_feature)))
        species_feature = self.species_dropout(F.selu(self.species_layer_two(species_feature)))
        x = torch.cat([lysate, cell, x, species_feature], dim=1)
        x = self.zero_dropout(self.batch_norm0(F.selu(self.zero_layer(x))))
        x = self.first_dropout(self.batch_norm1(F.selu(self.first_layer(x))))
        x = self.second_dropout(self.batch_norm2(F.selu(self.second_layer(x))))
        x = self.third_dropout(self.batch_norm3(F.selu(self.third_layer(x))))
        tm = self.seventh_layer(x)
        return tm

def run_deepstabp_inference(embeddings_prott5, ogt_values, lysate_type, checkpoint_path, device):
    """
    lysate_type: 'Lysate' or 'Cell'
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict']
    
    # Remove 'model.' prefix if present (though PL ckpt should have exactly the state keys matching DeepSTABp class)
    # Our inspection showed keys like 'zero_layer.weight' so it matches exactly.
    model = deepSTAPpMLP(dropout=0.1, learning_rate=0.01, batch_size=25)
    model.load_state_dict(state_dict)
    model = model.eval().to(device)
    
    n_seqs = len(embeddings_prott5)
    
    # Normalize OGT (species feature)
    # species = (species-30.44167)/(97.4167-30.44167)
    ogt_norm = (np.array(ogt_values) - 30.44167) / (97.4167 - 30.44167)
    ogt_norm = torch.tensor(ogt_norm, dtype=torch.float32).to(device)
    
    # Set lysate/cell indicators
    if lysate_type == 'Lysate':
        lysate_tensor = torch.ones(n_seqs, dtype=torch.float32).to(device)
        cell_tensor = torch.zeros(n_seqs, dtype=torch.float32).to(device)
    else:
        lysate_tensor = torch.zeros(n_seqs, dtype=torch.float32).to(device)
        cell_tensor = torch.ones(n_seqs, dtype=torch.float32).to(device)
        
    embeddings_prott5 = embeddings_prott5.to(device)
    
    preds_all = []
    batch_size = 128
    with torch.no_grad():
        for i in range(0, n_seqs, batch_size):
            x_batch = embeddings_prott5[i:i+batch_size]
            ogt_batch = ogt_norm[i:i+batch_size]
            lys_batch = lysate_tensor[i:i+batch_size]
            cell_batch = cell_tensor[i:i+batch_size]
            
            tm_preds = model(x_batch, ogt_batch, lys_batch, cell_batch)
            tm_preds = tm_preds.flatten().cpu().numpy()
            
            # Denormalize
            # tm_prediction = tm_prediction*(97.4166905791789-30.441673997070385)+30.441673997070385
            tm_preds = tm_preds * (97.4166905791789 - 30.441673997070385) + 30.441673997070385
            preds_all.append(tm_preds)
            
    return np.concatenate(preds_all)

def run_thermoformer_inference(sequences, model_name, device, batch_size=32):
    tokenizer = ThermoFormerTokenizer()
    model = ThermoFormer.from_pretrained(model_name)
    model = model.eval().to(device)
    
    # Truncate to 2048 to prevent OOM
    seqs_trunc = [s[:2048] for s in sequences]
    
    preds_all = []
    for i in tqdm(range(0, len(seqs_trunc), batch_size)):
        batch = seqs_trunc[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        preds_all.extend(outputs.predicted_values.cpu().numpy().flatten().tolist())
        
    return np.array(preds_all)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running DeepSTABp and ThermoFormer inference on device: {device}")
    
    # Check baseline predictions exists
    baseline_path = os.path.join(PROJECT_ROOT, "new_data/baseline_predictions.pt")
    if not os.path.exists(baseline_path):
        print(f"CRITICAL ERROR: {baseline_path} does not exist. Run run_baselines_inference.py first!")
        sys.exit(1)
        
    baselines = torch.load(baseline_path, map_location='cpu', weights_only=False)
    print("Loaded baseline predictions file.")
    
    # ==========================================
    # 1. LOAD TEST DATASETS
    # ==========================================
    
    # A. ProThermDB validation
    print("\nLoading ProThermDB validation sequences...")
    protherm_csv = os.path.join(PROJECT_ROOT, 'new_data/prothermdb_validation.csv')
    protherm_fasta = os.path.join(PROJECT_ROOT, 'new_data/prothermdb_validation.fasta')
    protherm_with_ogt_csv = os.path.join(PROJECT_ROOT, 'data/training_data/raw/prothermdb_validation_with_ogt.csv')
    prott5_data_path = os.path.join(PROJECT_ROOT, "new_data/prepared_data_v5_prott5.pt")
    
    df_p = pd.read_csv(protherm_csv)
    protherm_dict = {str(row['UniProt_ID']): float(row['Tm']) for _, row in df_p.iterrows() if not np.isnan(row['Tm'])}
    
    df_p_ogt = pd.read_csv(protherm_with_ogt_csv)
    protherm_ogt_dict = {str(row['UniProt_ID']): float(row['OGT']) for _, row in df_p_ogt.iterrows() if not np.isnan(row['OGT'])}
    
    protherm_seqs = []
    protherm_ogts = []
    for record in SeqIO.parse(protherm_fasta, 'fasta'):
        seq = str(record.seq)
        uid = record.id.split('|')[0]
        if uid in protherm_dict:
            protherm_seqs.append(seq)
            protherm_ogts.append(protherm_ogt_dict.get(uid, 37.0))
            
    # Load ProThermDB ProtT5 embeddings
    d_prott5 = torch.load(prott5_data_path, map_location='cpu', weights_only=False)
    protherm_prott5_embs = d_prott5['test_tm']['embeddings']
    print(f"Loaded {len(protherm_seqs)} ProThermDB sequences. Embeddings shape: {protherm_prott5_embs.shape}")
    
    # B. FireProt holdout
    print("\nLoading FireProt holdout sequences...")
    fireprot_path = os.path.join(PROJECT_ROOT, 'experiments/src/data/fireprot_holdout_prott5.pt')
    d_fireprot = torch.load(fireprot_path, map_location='cpu', weights_only=False)
    fireprot_seqs = d_fireprot['sequences']
    fireprot_prott5_embs = d_fireprot['embeddings_prott5']
    
    # Map FireProt sequences to UniProt KB IDs and lookup OGT
    base_dir_v7 = os.path.join(PROJECT_ROOT, "experiments/src/training/v7_transfer")
    uids_fireprot = map_fireprot_to_uniprot_ids(fireprot_seqs, base_dir_v7)
    
    with open(os.path.join(PROJECT_ROOT, "data/cleaner_data/tm_ogt_lookup.json")) as f:
        ogt_lookup = json.load(f)
        
    fireprot_ogts = []
    for uid in uids_fireprot:
        ogt_val = None
        if uid:
            ogt_info = ogt_lookup.get(uid, {})
            if ogt_info.get("source") == "known" and "ogt" in ogt_info:
                ogt_val = float(ogt_info["ogt"])
        if ogt_val is None:
            ogt_val = 37.0
        fireprot_ogts.append(ogt_val)
        
    print(f"Loaded {len(fireprot_seqs)} FireProt sequences. Embeddings shape: {fireprot_prott5_embs.shape}")
    
    # ==========================================
    # 2. RUN DEEPSTABP INFERENCE
    # ==========================================
    print("\n=== Running DeepSTABp Inference ===")
    deepstabp_ckpt = os.path.join(PROJECT_ROOT, "benchmark_models/DeepSTABp/src/Api/trained_model/b25_sampled_10k_tuned_2_d01/checkpoints/epoch=1-step=2316.ckpt")
    
    print("Predicting DeepSTABp on ProThermDB...")
    deepstabp_protherm_preds = run_deepstabp_inference(protherm_prott5_embs, protherm_ogts, 'Lysate', deepstabp_ckpt, device)
    
    print("Predicting DeepSTABp on FireProt...")
    deepstabp_fireprot_preds = run_deepstabp_inference(fireprot_prott5_embs, fireprot_ogts, 'Lysate', deepstabp_ckpt, device)
    
    # ==========================================
    # 3. RUN THERMOFORMER INFERENCE
    # ==========================================
    print("\n=== Running ThermoFormer Inference ===")
    thermoformer_model_name = "GinnM/ThermoFormer"
    
    print("Predicting ThermoFormer on ProThermDB (this may take a minute)...")
    thermoformer_protherm_preds = run_thermoformer_inference(protherm_seqs, thermoformer_model_name, device)
    
    print("Predicting ThermoFormer on FireProt...")
    thermoformer_fireprot_preds = run_thermoformer_inference(fireprot_seqs, thermoformer_model_name, device)
    
    # ==========================================
    # 4. SAVE MERGED PREDICTIONS
    # ==========================================
    print("\nMerging predictions and saving...")
    
    # Add to protherm dict
    baselines['protherm']['deepstabp'] = deepstabp_protherm_preds
    baselines['protherm']['thermoformer'] = thermoformer_protherm_preds
    
    # Add to fireprot dict
    baselines['fireprot']['deepstabp'] = deepstabp_fireprot_preds
    baselines['fireprot']['thermoformer'] = thermoformer_fireprot_preds
    
    torch.save(baselines, baseline_path)
    print(f"Successfully saved all baseline predictions (TemBERTure, ESMStabP, DeepSTABp, ThermoFormer) to {baseline_path}")

if __name__ == "__main__":
    main()
