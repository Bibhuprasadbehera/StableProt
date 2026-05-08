import os
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr
from model import MultiHead_TmPredictor
from config import CONFIG

def evaluate_model(model_path, data_path="../prepared_data_v2.pt"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        print("Data file not found. Have you run prepare_data_v2.py?")
        return
        
    data = torch.load(data_path, weights_only=True)
    test_tm = data['test_tm'] # ProThermDB
    train_tm = data['train_tm'] # for denormalization
    
    x = test_tm['embeddings'].to(device)
    y_true = test_tm['labels'].numpy()
    
    # Load model
    model = MultiHead_TmPredictor(
        input_size=CONFIG['input_size'],
        hidden1=CONFIG['hidden_size_1'],
        hidden2=CONFIG['hidden_size_2'],
        dropout1=CONFIG['dropout_1'],
        dropout2=CONFIG['dropout_2']
    ).to(device)
    
    print(f"Loading weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    tm_mean = train_tm['labels'].mean().item()
    tm_std = train_tm['labels'].std().item()
    
    print("Predicting...")
    with torch.no_grad():
        preds = model(x, head='tm').cpu()
        if CONFIG['target_normalization']:
            preds = preds * tm_std + tm_mean
            
    y_pred = preds.numpy()
    
    # Regression Metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    pcc, _ = pearsonr(y_true, y_pred)
    
    print(f"\n--- Regression Metrics on ProThermDB ---")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²:   {r2:.3f}")
    print(f"PCC:  {pcc:.3f}")
    
    # Binary Classification Metrics
    thresholds = list(range(0, 101, 5))
    print(f"\n--- Binary Classification Metrics ---")
    print(f"{'Threshold':<10} | {'AUC':<6} | {'F1':<6} | {'MCC':<6}")
    print("-" * 38)
    
    for t in thresholds:
        y_true_bin = (y_true >= t).astype(int)
        y_pred_bin = (y_pred >= t).astype(int)
        
        # We need at least one positive and one negative sample to compute AUC
        if len(np.unique(y_true_bin)) > 1:
            auc = roc_auc_score(y_true_bin, y_pred)
        else:
            auc = float('nan')
            
        f1 = f1_score(y_true_bin, y_pred_bin)
        mcc = matthews_corrcoef(y_true_bin, y_pred_bin)
        
        print(f"{t}°C{' '*(8-len(str(t)))} | {auc:.3f} | {f1:.3f} | {mcc:.3f}")

if __name__ == "__main__":
    if os.path.exists("results/best_model.pth"):
        evaluate_model("results/best_model.pth")
    else:
        print("No trained model found at results/best_model.pth!")
