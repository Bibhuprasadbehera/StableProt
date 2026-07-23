import os
import json
import matplotlib.pyplot as plt
import numpy as np

def main():
    json_path = "data/emergent_benchmarks/evaluation_results.json"
    if not os.path.exists(json_path):
        print(f"Results file {json_path} not found!")
        return

    with open(json_path, 'r') as fh:
        results = json.load(fh)

    tasks_mapping = {
        "HumanPPI (Acc)": "Task 2 (HumanPPI Accuracy)",
        "DeepLoc (Acc)": "Task 3 (DeepLoc cls2 Accuracy)",
        "eSOL (R2)": "Task 4 (eSOL R2)",
        "EC-1.x.x (Acc)": "Task 5 (EC-1.x.x.x Binary Accuracy)",
        "CB513 (Res-Acc)": "Task 6 (CB513 dssp3 Residue Accuracy)",
        "SCOP (Acc)": "Task 7 (SCOP Accuracy)"
    }

    rep_levels = [
        "Composition",
        "Raw SaProt",
        "ESM-2 (650M)",
        "ProtT5-XL",
        "StableProt-Tm MLP",
        "StableProt-OGT MLP",
        "StableProt-Combined",
        "StableProt-Predictions"
    ]

    # Premium color palette
    colors = [
        '#7F8C8D',  # Composition (Gray)
        '#3498DB',  # Raw SaProt (Blue)
        '#2ECC71',  # ESM-2 (Green)
        '#F1C40F',  # ProtT5 (Yellow)
        '#E67E22',  # StableProt-Tm MLP (Orange)
        '#E74C3C',  # StableProt-OGT MLP (Red)
        '#9B59B6',  # StableProt-Combined (Purple)
        '#1ABC9C'   # StableProt-Predictions (Teal)
    ]

    task_names = list(tasks_mapping.keys())
    data = {rep: [] for rep in rep_levels}

    for task_disp, task_key in tasks_mapping.items():
        task_data = results[task_key]
        for rep in rep_levels:
            if rep in task_data:
                scores = []
                if "Linear Probe" in task_data[rep]:
                    scores.append(task_data[rep]["Linear Probe"])
                if "MLP Probe" in task_data[rep]:
                    scores.append(task_data[rep]["MLP Probe"])
                best_score = max(scores) if scores else 0.0
                data[rep].append(best_score)
            else:
                data[rep].append(0.0)

    # Plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)

    x = np.arange(len(task_names))
    width = 0.10

    for i, rep in enumerate(rep_levels):
        offset = (i - 3.5) * width
        ax.bar(x + offset, data[rep], width, label=rep, color=colors[i], edgecolor='black', linewidth=0.5, alpha=0.9)

    ax.set_ylabel('Best Probe Performance Metric', fontsize=12, fontweight='bold')
    ax.set_title('StableProt vs Baselines: Emergent Property Evaluation Across Downstream Tasks', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(task_names, fontsize=11, fontweight='bold')
    ax.legend(frameon=True, facecolor='white', edgecolor='gray', framealpha=0.9, fontsize=10, loc='upper right')
    
    # Adjust limits and grid
    ax.set_ylim(-0.1, 1.05)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_out = "data/emergent_benchmarks/emergent_benchmark_results.png"
    plt.savefig(plot_out)
    print(f"Emergent benchmark results plot saved to {plot_out}")

if __name__ == "__main__":
    main()
