import os
import sys
import torch
import numpy as np
import pandas as pd
import hashlib
import joblib
from Bio import SeqIO
from tqdm import tqdm

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(PROJECT_ROOT)

# Import TemBERTure
sys.path.append(os.path.join(PROJECT_ROOT, "TemBERTure_repo/temBERTure"))
from temBERTure import TemBERTure

def load_cached_embedding(seq, cache_dir):
    seq_trunc = seq[:1500]
    h = hashlib.sha256(seq_trunc.encode()).hexdigest()
    fpath = os.path.join(cache_dir, f'mean_{h}.pt')
    if os.path.exists(fpath):
        data = torch.load(fpath, map_location='cpu')
        emb = data.get('mean_representations', None)
        if emb is not None:
            return emb if isinstance(emb, torch.Tensor) else torch.tensor(emb)
    return None

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running baseline inference on device: {device}")

    # ==========================================
    # 1. PREPARE TEST SEQUENCES
    # ==========================================
    
    # A. ProThermDB validation sequences
    print("\nPreparing ProThermDB validation sequences...")
    protherm_csv = os.path.join(PROJECT_ROOT, 'new_data/prothermdb_validation.csv')
    protherm_fasta = os.path.join(PROJECT_ROOT, 'new_data/prothermdb_validation.fasta')
    cache_dir = os.path.join(PROJECT_ROOT, 'experiments/embeddings_cache')
    
    df_p = pd.read_csv(protherm_csv)
    protherm_dict = {}
    for _, row in df_p.iterrows():
        rid = str(row['UniProt_ID'])
        tm = float(row['Tm'])
        if not np.isnan(tm):
            protherm_dict[rid] = tm

    protherm_seqs = []
    protherm_tms = []
    for record in SeqIO.parse(protherm_fasta, 'fasta'):
        seq = str(record.seq)
        emb = load_cached_embedding(seq, cache_dir)
        if emb is None:
            continue
        uid = record.id.split('|')[0]
        if uid in protherm_dict:
            protherm_seqs.append(seq)
            protherm_tms.append(protherm_dict[uid])
            
    print(f"Loaded {len(protherm_seqs)} ProThermDB validation sequences.")

    # B. FireProt holdout sequences
    print("\nPreparing FireProt holdout sequences...")
    fireprot_path = os.path.join(PROJECT_ROOT, 'experiments/src/data/fireprot_holdout_prott5.pt')
    d_fireprot = torch.load(fireprot_path, map_location='cpu')
    fireprot_seqs = d_fireprot['sequences']
    fireprot_tms = d_fireprot['temperatures'].numpy() if hasattr(d_fireprot['temperatures'], 'numpy') else np.array(d_fireprot['temperatures'])
    print(f"Loaded {len(fireprot_seqs)} FireProt holdout sequences.")

    # ==========================================
    # 2. RUN TEMBERTURE INFERENCE
    # ==========================================
    print("\n=== Running TemBERTure Inference ===")
    temberture_models = []
    for r in [1, 2, 3]:
        adapter_path = os.path.join(PROJECT_ROOT, f"TemBERTure_repo/temBERTure/temBERTure_TM/replica{r}/")
        print(f"Loading TemBERTure replica {r} from {adapter_path}...")
        model = TemBERTure(adapter_path=adapter_path, device=device, task='regression')
        temberture_models.append(model)
        
    def predict_temberture(seqs):
        preds_all = []
        for model in temberture_models:
            # We can predict in one go or chunked. predict handles batching internally.
            preds = model.predict(seqs)
            preds_all.append(preds)
        # Average across replicas
        return np.mean(preds_all, axis=0)

    print("Evaluating TemBERTure on ProThermDB...")
    temberture_protherm_preds = predict_temberture(protherm_seqs)
    
    print("Evaluating TemBERTure on FireProt...")
    temberture_fireprot_preds = predict_temberture(fireprot_seqs)

    # Clean up to free GPU memory
    del temberture_models
    torch.cuda.empty_cache()

    # ==========================================
    # 3. RUN ESMSTABP INFERENCE
    # ==========================================
    print("\n=== Running ESMStabP Inference ===")
    import esm
    
    print("Loading ESMStabP model (Model 1: sequence only)...")
    esmstabp_model_path = os.path.join(PROJECT_ROOT, "ESMStabP_repo/Models/1.joblib")
    esmstabp_rf = joblib.load(esmstabp_model_path)
    
    print("Loading ESM-2 650M model for embeddings...")
    model_esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model_esm = model_esm.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    REPR_LAYER = 33

    def get_esm_embeddings(seqs):
        embs = []
        for seq in tqdm(seqs):
            seq_trunc = seq[:1022]
            data = [("target", seq_trunc)]
            _, _, tokens = batch_converter(data)
            tokens = tokens.to(device)
            with torch.no_grad():
                results = model_esm(tokens, repr_layers=[REPR_LAYER], return_contacts=False)
                token_reps = results["representations"][REPR_LAYER][0, 1 : len(seq_trunc) + 1]
                emb = token_reps.mean(0).cpu().numpy()
            embs.append(emb)
        return np.array(embs)

    print("Extracting ESM-2 embeddings for ProThermDB...")
    protherm_embs = get_esm_embeddings(protherm_seqs)
    print("Predicting ESMStabP on ProThermDB...")
    esmstabp_protherm_preds = esmstabp_rf.predict(protherm_embs)

    print("Extracting ESM-2 embeddings for FireProt...")
    fireprot_embs = get_esm_embeddings(fireprot_seqs)
    print("Predicting ESMStabP on FireProt...")
    esmstabp_fireprot_preds = esmstabp_rf.predict(fireprot_embs)

    # ==========================================
    # 4. SAVE PREDICTIONS
    # ==========================================
    print("\nSaving predictions...")
    save_data = {
        'protherm': {
            'y_true': np.array(protherm_tms),
            'sequences': protherm_seqs,
            'temberture': temberture_protherm_preds,
            'esmstabp': esmstabp_protherm_preds
        },
        'fireprot': {
            'y_true': np.array(fireprot_tms),
            'sequences': fireprot_seqs,
            'temberture': temberture_fireprot_preds,
            'esmstabp': esmstabp_fireprot_preds
        }
    }
    
    os.makedirs(os.path.join(PROJECT_ROOT, "new_data"), exist_ok=True)
    out_file = os.path.join(PROJECT_ROOT, "new_data/baseline_predictions.pt")
    torch.save(save_data, out_file)
    print(f"Successfully saved baseline predictions to {out_file}")

if __name__ == "__main__":
    main()
