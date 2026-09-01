#!/usr/bin/env python3
import os
import re
import base64
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
PLOTS_DIR = PROJECT_ROOT / "paper" / "writeup" / "plots"
TABLES_DIR = PROJECT_ROOT / "paper" / "writeup" / "tables"
OUTPUT_HTML = PROJECT_ROOT / "paper" / "writeup" / "stableprot_v9_comprehensive_report.html"

def md_table_to_html(md_path):
    if not md_path.exists():
        return f"<p class='error'>Table file {md_path.name} not found.</p>"
    
    with open(md_path, 'r') as f:
        lines = f.readlines()
        
    html_lines = ["<div class='table-container'>", "<table>"]
    in_tbody = False
    
    for line in lines:
        line = line.strip()
        if not line.startswith('|'):
            continue
        # Split cell contents
        cells = [c.strip() for c in line.split('|')[1:-1]]
        
        # Skip markdown table divider line (e.g. |---|---|)
        if all(re.match(r'^:?-+:?$', c) for c in cells):
            continue
            
        if not in_tbody:
            html_lines.append("<thead><tr>")
            for cell in cells:
                html_lines.append(f"<th>{cell}</th>")
            html_lines.append("</tr></thead>")
            html_lines.append("<tbody>")
            in_tbody = True
        else:
            row_class = ""
            if any("StableProt V9" in cell for cell in cells):
                row_class = " class='highlight-row'"
            html_lines.append(f"<tr{row_class}>")
            for cell in cells:
                # Basic markdown formatting in cells
                cell_html = cell.replace('**', '<strong>').replace('__', '<strong>')
                html_lines.append(f"<td>{cell_html}</td>")
            html_lines.append("</tr>")
            
    if in_tbody:
        html_lines.append("</tbody>")
    html_lines.append("</table>")
    html_lines.append("</div>")
    return "\n".join(html_lines)

def png_to_base64(png_path):
    if not png_path.exists():
        print(f"Warning: Image {png_path.name} not found.")
        return ""
    with open(png_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded_string}"

def main():
    print("Building comprehensive self-contained HTML report...")
    
    # Core Regression Plots
    img_overall_metrics = png_to_base64(PLOTS_DIR / "overall_metrics_barplot.png")
    img_scatter_grid = png_to_base64(PLOTS_DIR / "comparative_scatter_grid.png")
    img_error_violins = png_to_base64(PLOTS_DIR / "error_distribution_violins.png")
    img_scatter_protherm = png_to_base64(PLOTS_DIR / "scatter_grid_prothermdb.png")
    img_scatter_fireprot = png_to_base64(PLOTS_DIR / "scatter_grid_fireprotdb.png")
    
    # Calibration
    img_cal_reliability = png_to_base64(PLOTS_DIR / "calibration_reliability_diagram.png")
    img_cal_stratified = png_to_base64(PLOTS_DIR / "calibration_stratified_temp.png")
    
    # OOD Generalization
    img_ood_cluster = png_to_base64(PLOTS_DIR / "cluster_ood_generalization.png")
    img_ood_species = png_to_base64(PLOTS_DIR / "cross_species_generalization.png")
    img_ood_brenda = png_to_base64(PLOTS_DIR / "ood_brenda_ogt.png")
    
    # Temp-wise profiles
    img_temp_protherm = png_to_base64(PLOTS_DIR / "temp_wise_protherm.png")
    img_temp_fireprot = png_to_base64(PLOTS_DIR / "temp_wise_fireprot.png")
    img_temp_int_ogt = png_to_base64(PLOTS_DIR / "internal_temp_wise_ogt.png")
    img_temp_ext_ogt = png_to_base64(PLOTS_DIR / "external_temp_wise_ogt.png")
    img_per_bin_mae = png_to_base64(PLOTS_DIR / "per_bin_mae_comparison.png")
    
    # Biophysical & Mutations
    img_mut_deltatm = png_to_base64(PLOTS_DIR / "mutation_deltatm_scatter.png")
    img_spurs_megascale = png_to_base64(PLOTS_DIR / "spurs_megascale_scatter.png")
    
    # Survival & Decoupling
    img_survival_metrics = png_to_base64(PLOTS_DIR / "survival_classification_metrics.png")
    img_roc_survival = png_to_base64(PLOTS_DIR / "roc_curves_survival_60c.png")
    img_grad_interference = png_to_base64(PLOTS_DIR / "gradient_interference_histogram.png")
    
    # Tables
    tab_protherm = md_table_to_html(TABLES_DIR / "table1_prothermdb.md")
    tab_fireprot = md_table_to_html(TABLES_DIR / "table3_fireprot.md")
    tab_ogt_brenda = md_table_to_html(TABLES_DIR / "table_ood_brenda_ogt.md")
    
    tab_temp_protherm = md_table_to_html(TABLES_DIR / "temp_wise_protherm.md")
    tab_temp_fireprot = md_table_to_html(TABLES_DIR / "temp_wise_fireprot.md")
    tab_temp_int_ogt = md_table_to_html(TABLES_DIR / "internal_temp_wise_ogt.md")
    tab_temp_ext_ogt = md_table_to_html(TABLES_DIR / "external_temp_wise_ogt.md")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>StableProt V9: Comprehensive Model Evaluation Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #080c14;
            --card-bg: rgba(17, 24, 39, 0.7);
            --card-hover-border: rgba(16, 185, 129, 0.4);
            --text-color: #f3f4f6;
            --text-muted: #9ca3af;
            --accent-primary: #10b981;
            --accent-secondary: #3b82f6;
            --accent-glow: rgba(16, 185, 129, 0.15);
            --border-color: rgba(255, 255, 255, 0.08);
            --shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
            --gradient: linear-gradient(135deg, #10b981 0%, #3b82f6 100%);
            --font-display: 'Outfit', sans-serif;
            --font-sans: 'Plus Jakarta Sans', sans-serif;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: var(--font-sans);
            background-color: var(--bg-color);
            background-image: 
                radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.08) 0px, transparent 50%),
                radial-gradient(at 50% 0%, rgba(16, 185, 129, 0.05) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(99, 102, 241, 0.05) 0px, transparent 50%);
            background-attachment: fixed;
            color: var(--text-color);
            line-height: 1.6;
            padding: 0;
        }}

        header {{
            position: relative;
            background: radial-gradient(ellipse at top, rgba(16, 185, 129, 0.12) 0%, rgba(8, 12, 20, 0) 70%);
            padding: 5rem 2rem 3rem 2rem;
            text-align: center;
            border-bottom: 1px solid var(--border-color);
            backdrop-filter: blur(10px);
        }}

        header::before {{
            content: '';
            position: absolute;
            bottom: -1px;
            left: 10%;
            width: 80%;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(16, 185, 129, 0.3), rgba(59, 130, 246, 0.3), transparent);
        }}

        h1 {{
            font-family: var(--font-display);
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 1rem;
            background: linear-gradient(135deg, #ffffff 30%, #a7f3d0 70%, #93c5fd 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.04em;
            text-shadow: 0 4px 20px rgba(16, 185, 129, 0.1);
        }}

        .subtitle {{
            color: var(--text-muted);
            font-size: 1.1rem;
            max-width: 850px;
            margin: 0 auto;
            font-weight: 300;
        }}

        .nav-tabs {{
            display: flex;
            justify-content: center;
            gap: 0.25rem;
            background-color: rgba(17, 24, 39, 0.85);
            padding: 0.4rem;
            border-radius: 9999px;
            max-width: 950px;
            margin: 2.5rem auto;
            border: 1px solid var(--border-color);
            box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.7);
            backdrop-filter: blur(20px);
            overflow-x: auto;
        }}

        .tab-btn {{
            background: none;
            border: none;
            color: var(--text-muted);
            padding: 0.8rem 1.6rem;
            font-size: 0.9rem;
            font-weight: 600;
            font-family: var(--font-sans);
            border-radius: 9999px;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
            white-space: nowrap;
        }}

        .tab-btn:hover {{
            color: #ffffff;
            background-color: rgba(255, 255, 255, 0.05);
        }}

        .tab-btn.active {{
            background: var(--gradient);
            color: #ffffff;
            box-shadow: 0 0 20px rgba(16, 185, 129, 0.4);
            transform: scale(1.02);
        }}

        main {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 2.5rem 6rem 2.5rem;
        }}

        .tab-content {{
            display: none;
            animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        }}

        .tab-content.active {{
            display: block;
        }}

        @keyframes slideUp {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .grid-container {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 3rem;
            margin-top: 2rem;
        }}

        @media (min-width: 1024px) {{
            .grid-container {{
                grid-template-columns: 1fr 1fr;
            }}
            .full-width {{
                grid-column: span 2;
            }}
        }}

        .card {{
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 2.5rem;
            box-shadow: var(--shadow);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
            position: relative;
            overflow: hidden;
        }}

        .card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.02) 0%, transparent 100%);
            pointer-events: none;
        }}

        .card:hover {{
            transform: translateY(-5px);
            border-color: var(--card-hover-border);
            box-shadow: 0 30px 60px -20px rgba(0, 0, 0, 0.8), 0 0 30px -5px var(--accent-glow);
        }}

        .card h2 {{
            font-family: var(--font-display);
            font-size: 1.6rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}

        .card p.description {{
            color: var(--text-muted);
            font-size: 0.95rem;
            margin-bottom: 2rem;
            font-weight: 300;
        }}

        .img-wrapper {{
            background-color: rgba(0, 0, 0, 0.3);
            border-radius: 14px;
            padding: 1.25rem;
            display: flex;
            justify-content: center;
            align-items: center;
            border: 1px solid rgba(255, 255, 255, 0.04);
            margin-bottom: 1.5rem;
            position: relative;
            overflow: hidden;
        }}

        .img-wrapper img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            max-height: 520px;
            object-fit: contain;
            transition: transform 0.5s cubic-bezier(0.16, 1, 0.3, 1);
        }}

        .img-wrapper:hover img {{
            transform: scale(1.015);
        }}

        /* Table Styling */
        .table-container {{
            overflow-x: auto;
            margin-top: 1rem;
            border-radius: 14px;
            border: 1px solid var(--border-color);
            background-color: rgba(0, 0, 0, 0.15);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
            text-align: left;
        }}

        th, td {{
            padding: 1rem 1.4rem;
            border-bottom: 1px solid var(--border-color);
        }}

        th {{
            background-color: rgba(255, 255, 255, 0.03);
            color: #ffffff;
            font-weight: 700;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.08em;
        }}

        tr:last-child td {{
            border-bottom: none;
        }}

        tr:hover td {{
            background-color: rgba(255, 255, 255, 0.02);
        }}

        /* Highlight V9 Rows */
        tr.highlight-row {{
            background-color: rgba(16, 185, 129, 0.08) !important;
        }}
        tr.highlight-row td {{
            border-bottom: 1px solid rgba(16, 185, 129, 0.25) !important;
            color: #ffffff !important;
            font-weight: 600;
        }}

        .badge {{
            display: inline-block;
            padding: 0.3rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            background-color: rgba(16, 185, 129, 0.12);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.2);
        }}

        .badge-blue {{
            background-color: rgba(59, 130, 246, 0.12);
            color: #60a5fa;
            border: 1px solid rgba(59, 130, 246, 0.2);
        }}

        .badge-purple {{
            background-color: rgba(139, 92, 246, 0.12);
            color: #a78bfa;
            border: 1px solid rgba(139, 92, 246, 0.2);
        }}

        /* Grid specific utilities */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.5rem;
            margin-top: 1.5rem;
        }}

        .metric-item {{
            background-color: rgba(0, 0, 0, 0.2);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 1.5rem;
            text-align: center;
            transition: all 0.3s ease;
        }}

        .metric-item:hover {{
            border-color: rgba(59, 130, 246, 0.3);
            background-color: rgba(59, 130, 246, 0.02);
        }}

        .metric-item .val {{
            font-size: 2rem;
            font-weight: 700;
            color: #ffffff;
            font-family: var(--font-display);
            background: linear-gradient(135deg, #ffffff 0%, #93c5fd 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .metric-item .lbl {{
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 0.4rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 600;
        }}

        .error {{
            color: #f87171;
            background-color: rgba(248, 113, 113, 0.08);
            border: 1px solid rgba(248, 113, 113, 0.2);
            padding: 1rem;
            border-radius: 8px;
            font-size: 0.9rem;
            margin: 1rem 0;
        }}
    </style>
</head>
<body>

    <header>
        <div class="badge">StableProt V9 Disjoint Release</div>
        <h1>StableProt V9 Analysis & Performance</h1>
        <p class="subtitle">Comprehensive evaluation metrics, uncertainty calibration, out-of-distribution generalization, and biophysical downstream validation results for the disjoint multi-head backbone architecture.</p>
    </header>

    <div class="nav-tabs">
        <button class="tab-btn active" onclick="openTab('regression')">Core Regression</button>
        <button class="tab-btn" onclick="openTab('calibration')">Calibration & Reliability</button>
        <button class="tab-btn" onclick="openTab('generalization')">OOD Generalization</button>
        <button class="tab-btn" onclick="openTab('downstream')">Biophysical & Mutations</button>
        <button class="tab-btn" onclick="openTab('safety')">Survival & Cosine Similarity</button>
    </div>

    <main>
        <!-- ── TAB: CORE REGRESSION ── -->
        <div id="regression" class="tab-content active">
            <div class="grid-container">
                <div class="card full-width">
                    <div>
                        <h2><span class="badge">Figure 1</span> Overall MAE Benchmarks (ProThermDB & FireProtDB)</h2>
                        <p class="description">Comparison of Mean Absolute Error (MAE, °C) across StableProt versions and state-of-the-art literature models on clean test partitions.</p>
                    </div>
                    <div class="img-wrapper">
                        <img src="{img_overall_metrics}" alt="Overall MAE Bar Plot">
                    </div>
                </div>

                <div class="card">
                    <div>
                        <h2><span class="badge badge-blue">Table</span> ProThermDB Regression Performance</h2>
                        <p class="description">Detailed comparative statistics (MAE, Pearson PCC, R², F1, AUC) evaluated on ProThermDB.</p>
                    </div>
                    {tab_protherm}
                </div>

                <div class="card">
                    <div>
                        <h2><span class="badge badge-blue">Table</span> FireProtDB Regression Performance</h2>
                        <p class="description">Detailed comparative statistics evaluated on the independent FireProtDB holdout.</p>
                    </div>
                    {tab_fireprot}
                </div>

                <div class="card full-width">
                    <div>
                        <h2><span class="badge">Figure 2</span> Comparative Scatter Grid (Predicted vs True)</h2>
                        <p class="description">Scatter plots demonstrating predicted vs true temperatures for all models on ProThermDB (left) and FireProtDB (right).</p>
                    </div>
                    <div class="img-wrapper">
                        <img src="{img_scatter_grid}" alt="Comparative Scatter Grid">
                    </div>
                </div>

                <div class="card">
                    <div>
                        <h2>ProThermDB Prediction Grid</h2>
                        <p class="description">Individual model scatter plot matrix on ProThermDB sequences.</p>
                    </div>
                    <div class="img-wrapper">
                        <img src="{img_scatter_protherm}" alt="ProThermDB Scatter Grid">
                    </div>
                </div>

                <div class="card">
                    <div>
                        <h2>FireProtDB Prediction Grid</h2>
                        <p class="description">Individual model scatter plot matrix on FireProtDB holdout sequences.</p>
                    </div>
                    <div class="img-wrapper">
                        <img src="{img_scatter_fireprot}" alt="FireProtDB Scatter Grid">
                    </div>
                </div>
            </div>
        </div>

        <!-- ── TAB: CALIBRATION & RELIABILITY ── -->
        <div id="calibration" class="tab-content">
            <div class="grid-container">
                <div class="card">
                    <div>
                        <h2><span class="badge">Figure 3</span> Reliability Diagram (Overall Calibration)</h2>
                        <p class="description">Expected coverage vs empirical coverage showing the quality of uncertainty bounds. The calibrated temperature-scaled model (T=2.8) achieves an ECE of 0.027.</p>
                    </div>
                    <div class="img-wrapper">
                        <img src="{img_cal_reliability}" alt="Reliability Diagram">
                    </div>
                </div>

                <div class="card">
                    <div>
                        <h2><span class="badge">Figure 4</span> Stratified Calibration Curves</h2>
                        <p class="description">Reliability curves stratified by thermal strata (mesophilic, thermophilic, hyperthermophilic) and 10°C temperature ranges.</p>
                    </div>
                    <div class="img-wrapper">
                        <img src="{img_cal_stratified}" alt="Stratified Calibration">
                    </div>
                </div>
            </div>
        </div>

        <!-- ── TAB: OOD GENERALIZATION ── -->
        <div id="generalization" class="tab-content">
            <div class="grid-container">
                <div class="card">
                    <div>
                        <h2><span class="badge">Figure 5</span> Sequence-Cluster Generalization (30% ID)</h2>
                        <p class="description">Model performance on sequence clusters binned by similarity. V9 disjoint architecture demonstrates robust generalizability to distant clusters compared to shared backbone models.</p>
                    </div>
                    <div class="img-wrapper">
                        <img src="{img_ood_cluster}" alt="Cluster Generalization">
                    </div>
                </div>

                <div class="card">
                    <div>
                        <h2><span class="badge">Figure 6</span> Cross-Species Stratification</h2>
                        <p class="description">Evaluating model performance across species of varying phylogenetic distance to the training distribution.</p>
                    </div>
                    <div class="img-wrapper">
                        <img src="{img_ood_species}" alt="Cross Species Generalization">
                    </div>
                </div>

                <div class="card full-width">
                    <div>
                        <h2><span class="badge">Figure 7</span> OOD BRENDA OGT Comparative Benchmark (N=525)</h2>
                        <p class="description">MAE Comparison on strictly decontaminated optimal growth temperature targets from BRENDA (<40% identity to training data).</p>
                    </div>
                    <div class="img-wrapper">
                        <img src="{img_ood_brenda}" alt="OOD BRENDA OGT">
                    </div>
                    {tab_ogt_brenda}
                </div>
            </div>
        </div>

        <!-- ── TAB: BIOPHYSICAL & MUTATIONS ── -->
        <div id="downstream" class="tab-content">
            <div class="grid-container">
                <div class="card">
                    <div>
                        <h2><span class="badge">Figure 8</span> Mutation ΔTm Prediction (ProThermDB Mutants)</h2>
                        <p class="description">Evaluating zero-shot prediction of mutant stability differences (ΔTm = Tm_mut - Tm_wt) using StableProt V9 predicted values.</p>
                    </div>
                    <div class="img-wrapper">
                        <img src="{img_mut_deltatm}" alt="Mutation Delta-Tm Scatter">
                    </div>
                </div>

                <div class="card">
                    <div>
                        <h2><span class="badge">Figure 9</span> Megascale Stability Generalization (SPURS)</h2>
                        <p class="description">Correlation with empirical high-throughput stability measurements from the SPURS megascale dataset.</p>
                    </div>
                    <div class="img-wrapper">
                        <img src="{img_spurs_megascale}" alt="SPURS Megascale Scatter">
                    </div>
                </div>

                <div class="card">
                    <div>
                        <h2><span class="badge">Figure 10</span> Per-Temperature-Bin MAE comparison</h2>
                        <p class="description">Dynamically computed binned MAE across the entire thermal spectrum for StableProt V9, PRIME, and ThermoFormer.</p>
                    </div>
                    <div class="img-wrapper">
                        <img src="{img_per_bin_mae}" alt="Per-Bin MAE Comparison">
                    </div>
                </div>

                <div class="card">
                    <div>
                        <h2>Temperature-Wise MAE Tables</h2>
                        <p class="description">Tabulated binned errors on internal BacDive (left) and external (right) datasets.</p>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                        <div>
                            <h4>Internal (BacDive)</h4>
                            {tab_temp_int_ogt}
                        </div>
                        <div>
                            <h4>External</h4>
                            {tab_temp_ext_ogt}
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- ── TAB: SAFETY & DECOUPLING ── -->
        <div id="safety" class="tab-content">
            <div class="grid-container">
                <div class="card">
                    <div>
                        <h2><span class="badge">Figure 11</span> Survival Classification Metrics</h2>
                        <p class="description">Model performance on classification tasks predicting whether a protein can survive above a threshold temperature.</p>
                    </div>
                    <div class="img-wrapper">
                        <img src="{img_survival_metrics}" alt="Survival Classification Metrics">
                    </div>
                </div>

                <div class="card">
                    <div>
                        <h2><span class="badge">Figure 12</span> ROC Curves (60°C Survival)</h2>
                        <p class="description">ROC Curves for predicting protein survival at 60°C on independent holdouts.</p>
                    </div>
                    <div class="img-wrapper">
                        <img src="{img_roc_survival}" alt="ROC Curves 60C Survival">
                    </div>
                </div>

                <div class="card full-width">
                    <div>
                        <h2><span class="badge">Figure 13</span> Shared Backbone Gradient Decoupling (Claim 5)</h2>
                        <p class="description">Histogram showing distribution of gradient cosine similarities. The disjoint backbone completely resolves gradient conflicts (similarity = 0.000) compared to shared models (mean similarity = -0.016).</p>
                    </div>
                    <div class="img-wrapper">
                        <img src="{img_grad_interference}" alt="Gradient Cosine Similarity Histogram">
                    </div>
                </div>
            </div>
        </div>
    </main>

    <script>
        function openTab(tabId) {{
            // Hide all tab contents
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(c => c.classList.remove('active'));

            // Remove active class from all buttons
            const buttons = document.querySelectorAll('.tab-btn');
            buttons.forEach(b => b.classList.remove('active'));

            // Show current tab content and set button active
            document.getElementById(tabId).classList.add('active');
            event.currentTarget.classList.add('active');
        }}
    </script>
</body>
</html>
"""
    
    with open(OUTPUT_HTML, "w") as f:
        f.write(html_content)
        
    print(f"Successfully generated self-contained report: {OUTPUT_HTML}")

if __name__ == "__main__":
    main()
