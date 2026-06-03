import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compute_binned_mae(y_true, predictions, bin_edges):
    num_bins = len(bin_edges) - 1
    bin_labels = [f"{bin_edges[i]}-{bin_edges[i+1]}" for i in range(num_bins)]
    
    bin_indices = np.digitize(y_true, bin_edges) - 1
    
    results = []
    
    for bin_idx in range(num_bins):
        mask = bin_indices == bin_idx
        count = np.sum(mask)
        
        if count == 0:
            continue
            
        bin_res = {
            'Bin': bin_labels[bin_idx],
            'Range': f"({bin_edges[bin_idx]}, {bin_edges[bin_idx+1]}]",
            'Count': int(count)
        }
        
        for name, y_pred in predictions.items():
            mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
            bin_res[name] = float(mae)
            
        results.append(bin_res)
        
    return pd.DataFrame(results)

def plot_temp_wise_mae(df_binned, title, save_path):
    plt.figure(figsize=(12, 6.5))
    sns.set_theme(style="whitegrid")
    
    # Exclude non-model columns
    model_cols = [col for col in df_binned.columns if col not in ['Bin', 'Range', 'Count']]
    
    # Harmonious color palette
    palette = sns.color_palette("husl", len(model_cols))
    
    # Plot line for each model
    for i, model_name in enumerate(model_cols):
        # We can make our key models (V6, V7) thicker
        linewidth = 3.0 if 'V6' in model_name or 'V7' in model_name else 1.5
        linestyle = '-' if 'V6' in model_name or 'V7' in model_name else '--'
        marker = 'o' if 'V6' in model_name or 'V7' in model_name else 's'
        
        plt.plot(
            df_binned['Bin'], 
            df_binned[model_name], 
            label=model_name,
            color=palette[i],
            linewidth=linewidth,
            linestyle=linestyle,
            marker=marker,
            markersize=6
        )
        
    plt.xlabel("Experimental Temperature Bin (°C)", fontsize=12, fontweight='bold')
    plt.ylabel("Mean Absolute Error (MAE) (°C)", fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True, fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved plot to {save_path}")

def df_to_markdown_simple(df):
    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        row_str = []
        for val in row:
            if isinstance(val, float):
                row_str.append(f"{val:.2f}")
            else:
                row_str.append(str(val))
        lines.append("| " + " | ".join(row_str) + " |")
    return "\n".join(lines)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    bin_edges = np.arange(0, 101, 10) # 0, 10, 20, ..., 100
    
    # 1. ProThermDB
    protherm_path = os.path.join(project_root, "new_data/protherm_evaluation_results.pt")
    if os.path.exists(protherm_path):
        print("Processing ProThermDB validation set...")
        data = torch.load(protherm_path, map_location='cpu', weights_only=False)
        y_true = np.array(data['y_true'])
        # Only include StableProt + external baselines
        show_models = ['StableProt', 'TemStaPro', 'TemBERTure', 'ESMStabP', 'DeepSTABp', 'ThermoFormer']
        predictions = {k: np.array(v) for k, v in data['predictions'].items() if k in show_models}
        
        df_protherm = compute_binned_mae(y_true, predictions, bin_edges)

        
        # Save as Markdown Table
        table_path = os.path.join(project_root, "paper/writeup/tables/temp_wise_protherm.md")
        os.makedirs(os.path.dirname(table_path), exist_ok=True)
        with open(table_path, "w") as f:
            f.write("# Temperature-Wise MAE Benchmark on ProThermDB Validation\n\n")
            f.write(df_to_markdown_simple(df_protherm))
        print(f"Saved ProThermDB table to {table_path}")
        
        # Plot
        plot_path = os.path.join(project_root, "paper/writeup/plots/temp_wise_protherm.png")
        plot_temp_wise_mae(df_protherm, "Temperature-Wise MAE Comparison on ProThermDB Validation", plot_path)
        
    # 2. FireProtDB
    fireprot_path = os.path.join(project_root, "new_data/fireprot_evaluation_results.pt")
    if os.path.exists(fireprot_path):
        print("\nProcessing FireProtDB holdout set...")
        data = torch.load(fireprot_path, map_location='cpu', weights_only=False)
        y_true = np.array(data['y_true'])
        show_models = ['StableProt', 'TemStaPro', 'TemBERTure', 'ESMStabP', 'DeepSTABp', 'ThermoFormer']
        predictions = {k: np.array(v) for k, v in data['predictions'].items() if k in show_models}
        
        df_fireprot = compute_binned_mae(y_true, predictions, bin_edges)

        
        # Save as Markdown Table
        table_path = os.path.join(project_root, "paper/writeup/tables/temp_wise_fireprot.md")
        with open(table_path, "w") as f:
            f.write("# Temperature-Wise MAE Benchmark on FireProtDB Holdout\n\n")
            f.write(df_to_markdown_simple(df_fireprot))
        print(f"Saved FireProtDB table to {table_path}")
        
        # Plot
        plot_path = os.path.join(project_root, "paper/writeup/plots/temp_wise_fireprot.png")
        plot_temp_wise_mae(df_fireprot, "Temperature-Wise MAE Comparison on FireProtDB Holdout", plot_path)

if __name__ == "__main__":
    main()
