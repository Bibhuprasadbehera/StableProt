import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import MultiHead_TmPredictor
from config import CONFIG
import pandas as pd

def generate_plots(model_path, data_path="../prepared_data_v2.pt"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
    
    print(f"Loading data from {data_path}...")
    data = torch.load(data_path, weights_only=True)
    test_tm = data['test_tm']
    
    x = test_tm['embeddings'].to(device)
    y_true = test_tm['labels'].numpy()
    sources = test_tm['source']
    
    model = MultiHead_TmPredictor(
        input_size=CONFIG['input_size'],
        hidden1=CONFIG['hidden_size_1'],
        hidden2=CONFIG['hidden_size_2'],
        dropout1=CONFIG['dropout_1'],
        dropout2=CONFIG['dropout_2']
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    tm_mean = data['train_tm']['labels'].mean().item()
    tm_std = data['train_tm']['labels'].std().item()
    
    with torch.no_grad():
        preds = model(x, head='tm').cpu()
        if CONFIG['target_normalization']:
            preds = preds * tm_std + tm_mean
            
    y_pred = preds.numpy()
    
    df = pd.DataFrame({
        'Actual Tm (°C)': y_true,
        'Predicted Tm (°C)': y_pred,
        'Source': sources,
        'Absolute Error': np.abs(y_true - y_pred)
    })
    
    # 1. Scatter Plot
    print("Generating Scatter Plot...")
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df, x='Actual Tm (°C)', y='Predicted Tm (°C)', hue='Source', alpha=0.6, s=15)
    
    # Diagonal line
    min_val = min(y_true.min(), y_pred.min()) - 5
    max_val = max(y_true.max(), y_pred.max()) + 5
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', zorder=1)
    
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.title("Actual vs Predicted Tm (Test Set)")
    plt.savefig(os.path.join(base_dir, "results/scatter_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Binned MAE Heatmap/Bar chart
    print("Generating Binned MAE Plot...")
    bins = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    labels = ['<20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '>90']
    df['Tm Bin'] = pd.cut(df['Actual Tm (°C)'], bins=bins, labels=labels)
    
    binned_mae = df.groupby('Tm Bin', observed=False)['Absolute Error'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=binned_mae, x='Tm Bin', y='Absolute Error', palette='viridis')
    plt.title("Mean Absolute Error by Temperature Bin")
    plt.ylabel("MAE (°C)")
    for i, v in enumerate(binned_mae['Absolute Error']):
        if not np.isnan(v):
            plt.text(i, v + 0.2, f"{v:.2f}", ha='center')
    plt.savefig(os.path.join(base_dir, "results/binned_mae.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Violin plot of Absolute Errors
    print("Generating Error Distribution Violin Plot...")
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x='Tm Bin', y='Absolute Error', inner="quartile", palette='muted')
    plt.title("Error Distribution by Temperature Bin")
    plt.ylabel("Absolute Error (°C)")
    plt.savefig(os.path.join(base_dir, "results/error_violin.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("All plots generated and saved to results/ directory!")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "results/best_model.pth")
    data_path = os.path.join(base_dir, "../../new_data/prepared_data_v2.pt")
    
    if os.path.exists(model_path):
        generate_plots(model_path, data_path)
    else:
        print(f"Model not found at {model_path}!")
