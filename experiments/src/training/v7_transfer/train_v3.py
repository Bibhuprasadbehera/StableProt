import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from config import CONFIG
from model import StableProtV7
from tqdm import tqdm

class TmDataset(Dataset):
    def __init__(self, ids, embeddings, labels, ogt_features, tm_features):
        self.ids = ids
        self.embeddings = embeddings
        self.labels = labels
        self.ogt_features = ogt_features
        self.tm_features = tm_features
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return (
            self.embeddings[idx], 
            self.labels[idx], 
            self.ogt_features[idx], 
            self.tm_features[idx]
        )

def get_temperature_weights(labels, bins=10, temp_range=(25, 100)):
    """Compute per-sample weights to balance temperature distribution during training."""
    labels_np = labels.numpy()
    bin_edges = np.linspace(temp_range[0], temp_range[1], bins + 1)
    bin_indices = np.digitize(labels_np, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, bins - 1)
    
    bin_counts = np.bincount(bin_indices, minlength=bins)
    # Calculate inverse frequency
    bin_weights = 1.0 / np.maximum(bin_counts, 1)
    # Normalize weights
    bin_weights = bin_weights / bin_weights.sum()
    
    sample_weights = bin_weights[bin_indices]
    # Convert to tensor
    return torch.tensor(sample_weights, dtype=torch.float64)

def precompute_features(ids, embeddings, ogt_lookup, tm_lookup, stage1_predictor, device):
    """Precompute OGT and TM features for all samples to avoid on-the-fly model calls."""
    print("Precomputing features...")
    ogt_features = []
    tm_features = []
    
    # Process in batches to avoid GPU OOM if predicting OGT
    batch_size = 512
    
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i+batch_size]
        batch_embs = embeddings[i:i+batch_size].to(device)
        
        batch_ogt = []
        batch_tm = []
        
        # We need to run predictions for IDs that lack known OGT
        needs_pred_indices = []
        needs_pred_embs = []
        
        for j, full_id in enumerate(batch_ids):
            uid = full_id.split("|")[0]
            # TM lookup
            tm_val = float(tm_lookup.get(uid, 0))
            batch_tm.append(tm_val)
            
            # OGT lookup
            ogt_info = ogt_lookup.get(uid, {})
            if ogt_info.get("source") == "known" and "ogt" in ogt_info:
                batch_ogt.append(float(ogt_info["ogt"]))
            else:
                # Mark for prediction fallback
                batch_ogt.append(None)
                needs_pred_indices.append(j)
                needs_pred_embs.append(batch_embs[j])
                
        # Run OGT prediction batch if needed
        if needs_pred_indices and stage1_predictor is not None:
            needs_pred_embs = torch.stack(needs_pred_embs)
            with torch.no_grad():
                preds = stage1_predictor(needs_pred_embs, stage='ogt').cpu().tolist()
            for idx, p_val in zip(needs_pred_indices, preds):
                batch_ogt[idx] = p_val
        elif needs_pred_indices:
            # Fallback if no predictor (e.g. stage 1 weights not found)
            for idx in needs_pred_indices:
                batch_ogt[idx] = 37.0 # default mesophilic
                
        ogt_features.extend(batch_ogt)
        tm_features.extend(batch_tm)
        
    return torch.tensor(ogt_features, dtype=torch.float32), torch.tensor(tm_features, dtype=torch.float32)

def train_seed_stage2(mode, seed, data, ogt_lookup, tm_lookup, device, save_dir, stage1_model_path=None):
    print("\n" + "="*50)
    print(f"Stage 2 Tm Training | Mode: {mode} | Seed {seed}")
    print("="*50)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Define configuration switches based on Mode
    use_ogt_feature = (mode in ['D3', 'D4'])
    use_tm_feature = (mode == 'D4')
    use_stratified_sampling = (mode in ['D2', 'D3', 'D4'])

    # 1. Load Stage 1 model to precompute predictions if needed
    stage1_predictor = None
    if stage1_model_path and os.path.exists(stage1_model_path):
        print(f"Loading Stage 1 predictor from {stage1_model_path} for OGT prediction fallback...")
        stage1_predictor = StableProtV7(
            emb_dim=CONFIG['input_size'],
            hidden=CONFIG['hidden_size'],
            bottleneck=CONFIG['bottleneck_size']
        ).to(device)
        stage1_predictor.load_state_dict(torch.load(stage1_model_path, map_location=device, weights_only=True))
        stage1_predictor.eval()
    else:
        print("WARNING: Stage 1 predictor not found. Will use default OGT fallback if lookup fails.")

    # 2. Precompute features for Train, Val, Test splits
    print("\n--- Precomputing train features ---")
    train_ogt, train_tm_feat = precompute_features(
        data['train_tm']['ids'], data['train_tm']['embeddings'], 
        ogt_lookup, tm_lookup, stage1_predictor, device
    )
    print("\n--- Precomputing val features ---")
    val_ogt, val_tm_feat = precompute_features(
        data['val_tm']['ids'], data['val_tm']['embeddings'], 
        ogt_lookup, tm_lookup, stage1_predictor, device
    )
    print("\n--- Precomputing test features ---")
    test_ogt, test_tm_feat = precompute_features(
        data['test_tm']['ids'], data['test_tm']['embeddings'], 
        ogt_lookup, tm_lookup, stage1_predictor, device
    )

    # 3. Create datasets and loaders
    train_dataset = TmDataset(
        data['train_tm']['ids'], data['train_tm']['embeddings'], data['train_tm']['labels'], 
        train_ogt, train_tm_feat
    )
    val_dataset = TmDataset(
        data['val_tm']['ids'], data['val_tm']['embeddings'], data['val_tm']['labels'], 
        val_ogt, val_tm_feat
    )
    test_dataset = TmDataset(
        data['test_tm']['ids'], data['test_tm']['embeddings'], data['test_tm']['labels'], 
        test_ogt, test_tm_feat
    )

    # Setup sampler
    if use_stratified_sampling:
        print("Using temperature-stratified sampler for training loader...")
        weights = get_temperature_weights(data['train_tm']['labels'])
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], sampler=sampler, drop_last=True)
    else:
        print("Using uniform random sampler for training loader...")
        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)

    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    # 4. Instantiate model
    model = StableProtV7(
        emb_dim=CONFIG['input_size'],
        hidden=CONFIG['hidden_size'],
        bottleneck=CONFIG['bottleneck_size'],
        dropout1=CONFIG['dropout_1'],
        dropout2=CONFIG['dropout_2'],
        use_ogt_feature=use_ogt_feature,
        use_tm_feature=use_tm_feature
    ).to(device)

    # Since we are not doing OGT backbone transfer (it was shown to hurt performance),
    # we initialize the backbone randomly and train everything from scratch!
    optimizer = optim.Adam(
        model.parameters(),
        # Use stage2_head_lr for training everything
        lr=CONFIG['stage2_head_lr'],
        weight_decay=CONFIG['weight_decay'] * 10.0 # increased regularization since dataset is smaller (29K)
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=CONFIG['lr_scheduler_patience'],
        factor=CONFIG['lr_scheduler_factor'],
        min_lr=1e-6
    )

    criterion = nn.HuberLoss(delta=CONFIG['huber_delta'])

    best_val_mae = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(save_dir, f'model_{mode}.pt')

    for epoch in range(CONFIG['stage2_epochs']):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['stage2_epochs']}")
        for batch_idx, (x, y, ogt_val, tm_val) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            ogt_val = ogt_val.to(device) if use_ogt_feature else None
            tm_val = tm_val.to(device) if use_tm_feature else None

            optimizer.zero_grad()
            pred = model(x, stage='tm', ogt_pred=ogt_val, tm_feat=tm_val)
            loss = criterion(pred, y)
            loss.backward()

            if CONFIG['grad_clip_max_norm'] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip_max_norm'])

            optimizer.step()
            train_loss += loss.item()

            if batch_idx % 20 == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Validation
        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for x, y, ogt_val, tm_val in val_loader:
                x, y = x.to(device), y.to(device)
                ogt_val = ogt_val.to(device) if use_ogt_feature else None
                tm_val = tm_val.to(device) if use_tm_feature else None

                pred = model(x, stage='tm', ogt_pred=ogt_val, tm_feat=tm_val)
                val_mae += torch.abs(pred - y).sum().item()

        val_mae /= len(val_loader.dataset)
        scheduler.step(val_mae)

        print(f"Epoch {epoch+1} - Val MAE: {val_mae:.4f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("Saved best model!")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['early_stopping_patience']:
                print("Early stopping triggered!")
                break

    # Load best model for testing
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    preds = []
    with torch.no_grad():
        for x, _, ogt_val, tm_val in test_loader:
            x = x.to(device)
            ogt_val = ogt_val.to(device) if use_ogt_feature else None
            tm_val = tm_val.to(device) if use_tm_feature else None

            p = model(x, stage='tm', ogt_pred=ogt_val, tm_feat=tm_val).cpu()
            preds.extend(p.tolist())

    # Save predictions
    test_labels = torch.cat([y for _, y, _, _ in test_loader])
    torch.save(
        {'y_true': test_labels, 'y_pred': torch.tensor(preds)},
        os.path.join(save_dir, f'predictions_{mode}.pt')
    )

    mae = np.mean(np.abs(np.array(preds) - test_labels.numpy()))
    print(f"Mode {mode} | Seed {seed} | Test MAE: {mae:.4f}")

    with open(os.path.join(save_dir, f'metrics_{mode}.json'), 'w') as f:
        json.dump({'mae': float(mae)}, f)

    return preds

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['D1', 'D2', 'D3', 'D4'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "../../../data/cleaner_data/prepared_data_v3.pt")
    ogt_lookup_path = os.path.join(base_dir, "../../../data/cleaner_data/tm_ogt_lookup.json")
    tm_lookup_path = os.path.join(base_dir, "../../../data/cleaner_data/tm_transmembrane.json")

    print("Loading dataset...")
    data = torch.load(data_path, map_location="cpu")
    
    print("Loading features lookup files...")
    with open(ogt_lookup_path) as f:
        ogt_lookup = json.load(f)
    with open(tm_lookup_path) as f:
        tm_lookup = json.load(f)

    results_dir = os.path.join(base_dir, 'results')
    ensemble_preds = []

    for seed in CONFIG['seeds']:
        seed_dir = os.path.join(results_dir, f"seed{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        
        # Load stage1 model path if exists
        stage1_model_path = os.path.join(seed_dir, 'model_stage1.pt')

        preds = train_seed_stage2(
            args.mode, seed, data, ogt_lookup, tm_lookup,
            device, seed_dir, stage1_model_path=stage1_model_path
        )
        if preds is not None:
            ensemble_preds.append(preds)

    # Ensemble predictions
    if ensemble_preds:
        ensemble_dir = os.path.join(results_dir, f"ensemble_{args.mode}")
        os.makedirs(ensemble_dir, exist_ok=True)
        mean_preds = np.mean(ensemble_preds, axis=0)
        
        test_tm_lbl = data['test_tm']['labels']
        mae = np.mean(np.abs(mean_preds - test_tm_lbl.numpy()))

        torch.save(
            {'y_true': test_tm_lbl, 'y_pred': torch.tensor(mean_preds)},
            os.path.join(ensemble_dir, 'predictions.pt')
        )

        with open(os.path.join(ensemble_dir, 'metrics.json'), 'w') as f:
            json.dump({'mae': float(mae)}, f)

        print(f"\n[Ensemble Mode {args.mode}] Test MAE: {mae:.4f}")

if __name__ == "__main__":
    main()
