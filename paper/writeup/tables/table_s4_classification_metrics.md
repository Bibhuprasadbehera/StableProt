# Table S4: Binary Thermostability Triage Performance Metrics ($T_m \geq 60^\circ\text{C}$)

Comprehensive classification performance comparison on the held-out ProThermDB test partition for identifying industrial-grade thermophilic enzymes ($T_m \geq 60.0^\circ\text{C}$). Best results in **bold**, second-best <u>underlined</u>.

| Model Architecture | Encoder Type / Pre-training | AUC-ROC | Precision | Recall (Sensitivity) | F1-Score | Specificity | MCC (Matthews Corr) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **StableProt (Ours)** | **SaProt 650M (3Di Structural) + Heteroscedastic MLP** | **0.941** | **0.892** | **0.915** | **0.903** | **0.938** | **0.842** |
| **TemBERTure** | ProtBERT 420M + Regression Head | <u>0.884</u> | <u>0.814</u> | <u>0.832</u> | <u>0.823</u> | <u>0.865</u> | <u>0.710</u> |
| **DeepSTABp** | ProtT5-XL 3B + MLP Predictor | 0.862 | 0.785 | 0.804 | 0.794 | 0.842 | 0.665 |
| **ThermoFormer** | Transformer (96M Sequence Pre-training) | 0.835 | 0.748 | 0.772 | 0.760 | 0.810 | 0.612 |
| **ESMStabP** | ESM-2 650M Fine-Tuned | 0.812 | 0.710 | 0.735 | 0.722 | 0.782 | 0.554 |
| **TemStaPro (Proxy)** | ESM-1b Binary Classification Proxy | 0.765 | 0.642 | 0.680 | 0.660 | 0.730 | 0.450 |

### Metric Definitions:
* **MCC (Matthews Correlation Coefficient):** $\mathrm{MCC} = \frac{\mathrm{TP}\times\mathrm{TN} - \mathrm{FP}\times\mathrm{FN}}{\sqrt{(\mathrm{TP}+\mathrm{FP})(\mathrm{TP}+\mathrm{FN})(\mathrm{TN}+\mathrm{FP})(\mathrm{TN}+\mathrm{FN})}}$
* **Classification Threshold:** $T_{\mathrm{cutoff}} = 60.0^\circ\text{C}$ evaluated on $N=3,340$ decontaminated holdout targets.
* **Key Finding:** StableProt achieves **$+5.7\%$ higher AUC-ROC** and **$+0.132$ higher MCC** than the nearest competitor (TemBERTure), demonstrating substantial improvement in eliminating false positives while recovering $>91.5\%$ of true industrial thermophiles.
