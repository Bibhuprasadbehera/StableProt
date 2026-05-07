import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

# A simple MLP similar to Head A
class SimpleMLP(nn.Module):
    def __init__(self, input_size=2560, hidden1=512, hidden2=256, dropout1=0.3, dropout2=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(hidden2, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_and_eval(X_train, y_train, X_val, y_val, device='cuda'):
    model = SimpleMLP().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    model.train()
    for epoch in range(15):  # Train for a few epochs
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        X_val_dev = X_val.to(device)
        preds = model(X_val_dev).cpu().numpy()
        mae = mean_absolute_error(y_val.numpy(), preds)
        
    return mae

def main():
    print("Layer Selection Experiment")
    
    # Check if the embeddings exist
    cache_dir = "esm2_embeddings_cache/subset_1k"
    if not os.path.exists(cache_dir):
        print(f"Error: {cache_dir} not found. Generate embeddings first.")
        return
        
    embeddings_list = []
    targets = []
    
    for file in os.listdir(cache_dir):
        if file.endswith(".pt") and "targets" not in file:
            path = os.path.join(cache_dir, file)
            emb = torch.load(path)
            # emb is shape (3, 2560) for layers [30, 33, 36]
            embeddings_list.append(emb)
    
    # We need the targets too
    # Load targets mapped from the fasta
    target_path = os.path.join(cache_dir, "targets.pt")
    if os.path.exists(target_path):
        targets = torch.load(target_path)
    else:
        print("Error: targets.pt not found. Run the subset generation script.")
        return
    
    embeddings_tensor = torch.stack(embeddings_list) # (N, 3, 2560)
    targets_tensor = torch.tensor(targets, dtype=torch.float32) # (N,)
    
    print(f"Loaded {len(targets_tensor)} samples.")
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings_tensor, targets_tensor, test_size=0.2, random_state=42
    )
    
    layers = [30, 33, 36]
    best_layer = None
    best_mae = float('inf')
    
    for i, layer in enumerate(layers):
        print(f"Training on embeddings from layer {layer}...")
        X_train_layer = X_train[:, i, :]
        X_val_layer = X_val[:, i, :]
        
        # Run multiple seeds for stability
        maes = []
        for seed in range(3):
            torch.manual_seed(seed)
            mae = train_and_eval(X_train_layer, y_train, X_val_layer, y_val)
            maes.append(mae)
            
        avg_mae = np.mean(maes)
        print(f"Layer {layer} MAE: {avg_mae:.4f} (std: {np.std(maes):.4f})")
        
        if avg_mae < best_mae:
            best_mae = avg_mae
            best_layer = layer
            
    print(f"\\nBest layer is {best_layer} with MAE: {best_mae:.4f}")
    
if __name__ == "__main__":
    main()
