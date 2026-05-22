import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, EXPERIMENTS_DIR)

from model import MLP_Regression
from config import CONFIG
from utils.data_utils import TemStaProDataset
from torch.utils.data import DataLoader

def create_regression_data_loaders(data_path, batch_size):
    print("Loading prepared data from: %s" % data_path)
    data = torch.load(data_path)
    
    # We use actual temperatures as targets
    train_dataset = TemStaProDataset(data['train_embeddings'], torch.tensor(data['train_temps'], dtype=torch.float32))
    val_dataset = TemStaProDataset(data['val_embeddings'], torch.tensor(data['val_temps'], dtype=torch.float32))
    test_dataset = TemStaProDataset(data['test_embeddings'], torch.tensor(data['test_temps'], dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, data['test_temps']

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device).float(), targets.to(device).float()
            outputs = model(inputs).squeeze(-1)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            
            all_preds.extend(outputs.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    avg_loss = total_loss / len(data_loader.dataset)
    
    # Calculate MAE as an additional metric
    preds = np.array(all_preds)
    targets = np.array(all_targets)
    mae = np.mean(np.abs(preds - targets))
    
    return avg_loss, mae

def train_model(train_loader, val_loader, seed, save_dir, device):
    print("\n" + "="*50)
    print("Training Regression Model (Seed %d)" % seed)
    print("="*50)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = MLP_Regression(
        input_size=CONFIG['input_size'],
        hidden_size_1=CONFIG['hidden_size_1'],
        hidden_size_2=CONFIG['hidden_size_2'],
        dropout_1=CONFIG['dropout_1'],
        dropout_2=CONFIG['dropout_2']
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(save_dir, 'model.pt')
    
    history = {'train_loss': [], 'val_loss': [], 'val_mae': []}

    for epoch in range(CONFIG['num_epochs']):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device).float(), targets.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss, val_mae = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)

        print("Epoch %2d/%2d | Train MSE: %.4f | Val MSE: %.4f | Val MAE: %.4f" % 
              (epoch+1, CONFIG['num_epochs'], train_loss, val_loss, val_mae))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("  -> Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['early_stopping_patience']:
                print("Early stopping triggered at epoch %d" % (epoch+1))
                break

    # Plot history
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train MSE')
    plt.plot(history['val_loss'], label='Val MSE')
    plt.legend()
    plt.title('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_mae'], label='Val MAE', color='green')
    plt.legend()
    plt.title('Mean Absolute Error')
    
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()

    return best_model_path

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None)
    args = parser.parse_args()

    data_path = args.data if args.data else os.path.join(EXPERIMENTS_DIR, CONFIG['data_path'])
    if not os.path.exists(data_path):
        data_path = os.path.join(EXPERIMENTS_DIR, 'prepared_data.pt')
        if not os.path.exists(data_path):
            print("ERROR: Could not find prepared data at %s" % data_path)
            sys.exit(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device: %s" % device)

    results_dir = os.path.join(SCRIPT_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)

    train_loader, val_loader, test_loader, test_temps = create_regression_data_loaders(
        data_path, CONFIG['batch_size']
    )

    ensemble_preds = []

    for seed in CONFIG['seeds']:
        seed_dir = os.path.join(results_dir, "seed%d" % seed)
        os.makedirs(seed_dir, exist_ok=True)

        best_model_path = train_model(train_loader, val_loader, seed, seed_dir, device)

        # Evaluate on test set
        model = MLP_Regression(
            input_size=CONFIG['input_size'],
            hidden_size_1=CONFIG['hidden_size_1'],
            hidden_size_2=CONFIG['hidden_size_2']
        ).to(device)
        model.load_state_dict(torch.load(best_model_path))
        
        model.eval()
        preds = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device).float()
                preds.extend(model.predict(inputs).cpu().tolist())
        
        ensemble_preds.append(preds)
        
        # Save predictions
        torch.save({
            'y_true': torch.tensor(test_temps),
            'y_pred': torch.tensor(preds)
        }, os.path.join(seed_dir, 'predictions.pt'))
        
        mae = np.mean(np.abs(np.array(preds) - np.array(test_temps)))
        with open(os.path.join(seed_dir, 'metrics.json'), 'w') as f:
            json.dump({'mae': mae}, f, indent=2)

    # Ensemble evaluation
    ensemble_dir = os.path.join(results_dir, "ensemble")
    os.makedirs(ensemble_dir, exist_ok=True)
    
    mean_preds = np.mean(ensemble_preds, axis=0)
    mae = np.mean(np.abs(mean_preds - np.array(test_temps)))
    
    print("\n" + "="*50)
    print("ENSEMBLE REGRESSION RESULTS")
    print("="*50)
    print("Overall Test MAE: %.4f °C" % mae)
    
    torch.save({
        'y_true': torch.tensor(test_temps),
        'y_pred': torch.tensor(mean_preds)
    }, os.path.join(ensemble_dir, 'predictions.pt'))
    
    with open(os.path.join(ensemble_dir, 'metrics.json'), 'w') as f:
        json.dump({'mae': mae}, f, indent=2)

if __name__ == '__main__':
    main()
