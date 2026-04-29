# Future Improvement Suggestions

These are additional strategies to try after comparing V1 (baseline) vs V2 (weighted + balanced + dropout). Implement these as **v3**, **v4**, etc. for systematic comparison.

---

## 1. Focal Loss (v3 candidate)

**Best for: extreme imbalance at thresholds ≥ 60°C**

Focal Loss automatically down-weights easy/well-classified examples and focuses training on hard, misclassified ones.

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()
```

- `gamma=2.0` → easy examples contribute ~100× less loss
- `alpha=0.25` → upweights minority class
- Especially useful at 65°C+ where positive class is <7%

---

## 2. SMOTE on Embeddings (v4 candidate)

Apply Synthetic Minority Over-sampling in the 1024-d embedding space:

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
```

- Install: `pip install imbalanced-learn`
- Apply **only on training set**, never on val/test
- Works because ProtT5 embeddings are continuous vectors where interpolation is meaningful

---

## 3. Regression Approach (v5 candidate — paradigm shift)

Train **one model** to predict OGT directly, instead of 9 binary classifiers:

```python
class MLP_Regression(nn.Module):
    def __init__(self, input_size=1024):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)  # Raw temperature output
        )

    def forward(self, x):
        return self.model(x)
```

**Advantages:**
- 1 model instead of 45 (9 thresholds × 5 seeds)
- Skew is less problematic (learns full temperature distribution)
- More informative: "predicted OGT = 72°C" vs "thermophilic = yes"
- Derive binary labels after: `predicted_temp >= threshold`
- Use `nn.HuberLoss(delta=5.0)` (robust to 100°C outliers)

---

## 4. Temperature-Aware Sampling

Instead of simple balanced sampling, weight samples by how close they are to the decision boundary:

```python
# Samples near the threshold boundary are more informative
def boundary_aware_weights(temps, threshold, sigma=5.0):
    distances = np.abs(np.array(temps) - threshold)
    weights = np.exp(-distances**2 / (2 * sigma**2))
    weights = weights / weights.sum()
    return weights
```

---

## 5. Threshold-Specific Strategies

Different thresholds have different imbalance levels. Use different strategies:

| Threshold | Imbalance | Strategy |
|-----------|-----------|----------|
| 40–50°C   | Moderate (3:1–4:1) | Weighted BCE + balanced sampling |
| 55–65°C   | High (6:1–14:1) | Focal Loss + SMOTE |
| 70–80°C   | Extreme (20:1–50:1) | Focal Loss + aggressive oversampling + lower LR |

---

## 6. Contrastive Learning / Embedding Fine-tuning

Fine-tune the ProtT5 embeddings (or a projection layer) with a contrastive loss that pulls thermophilic proteins closer together and pushes them away from mesophilic ones:

```python
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=1024, proj_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim)
        )

    def forward(self, x):
        return nn.functional.normalize(self.projection(x), dim=-1)
```

Use supervised contrastive loss (SupCon) to create better-separated embeddings, then train the classifier on these improved representations.

---

## 7. Learning Rate Scheduling

Add cosine annealing or reduce-on-plateau:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)
# OR
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
```

---

## 8. Label Smoothing

Soften labels to prevent overconfident predictions:

```python
# Instead of hard 0/1 labels:
smooth_labels = labels * (1 - 0.1) + 0.05  # 0→0.05, 1→0.95
```

---

## Implementation Priority

| Priority | Suggestion | Expected Impact | Effort |
|----------|-----------|----------------|--------|
| 🔴 High | Focal Loss | Major improvement at high thresholds | 30 min |
| 🔴 High | Regression model | Paradigm shift, potentially much better | 2-3 hrs |
| 🟡 Medium | SMOTE | Moderate improvement | 1 hr |
| 🟡 Medium | LR scheduling | Slightly better convergence | 15 min |
| 🟢 Low | Contrastive learning | Potentially large but experimental | 4+ hrs |
| 🟢 Low | Label smoothing | Minor improvement | 10 min |
| 🟢 Low | Boundary-aware sampling | Minor but interesting | 30 min |
