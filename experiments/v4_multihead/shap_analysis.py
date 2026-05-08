import os
import torch
import shap
import matplotlib.pyplot as plt
from model import MultiHead_TmPredictor
from config import CONFIG
import numpy as np

def run_shap_analysis(model_path, data_path="../prepared_data_v2.pt"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        print("Data file not found.")
        return
        
    data = torch.load(data_path, weights_only=True)
    
    # We will compute SHAP values on the ProThermDB test set
    x_test = data['test_tm']['embeddings'][:500].to(device) # Limit to 500 for speed
    
    # We need a background dataset for DeepExplainer
    # We take 100 random samples from the training set
    x_train = data['train_tm']['embeddings']
    idx = torch.randperm(len(x_train))[:100]
    background = x_train[idx].to(device)
    
    model = MultiHead_TmPredictor(
        input_size=CONFIG['input_size'],
        hidden1=CONFIG['hidden_size_1'],
        hidden2=CONFIG['hidden_size_2'],
        dropout1=CONFIG['dropout_1'],
        dropout2=CONFIG['dropout_2']
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    print("Setting up SHAP DeepExplainer on the backbone output...")
    # Since we want to explain the predictions, we can either explain raw ESM-2 (2560 dims)
    # or the hidden layer. Raw ESM-2 is better to see which embedding dimensions matter.
    
    # Wrapper to only return Tm predictions
    class TmHeadWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            return self.model(x, head='tm').unsqueeze(1)
            
    wrapped_model = TmHeadWrapper(model)
    
    explainer = shap.DeepExplainer(wrapped_model, background)
    
    print("Calculating SHAP values... This may take a few minutes.")
    shap_values = explainer.shap_values(x_test)
    
    print("Generating Beeswarm plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, x_test.cpu().numpy(), show=False)
    
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/shap_summary.png", bbox_inches='tight', dpi=300)
    print("Saved SHAP summary plot to results/shap_summary.png")

if __name__ == "__main__":
    if os.path.exists("results/best_model.pth"):
        run_shap_analysis("results/best_model.pth")
    else:
        print("Model not trained yet!")
