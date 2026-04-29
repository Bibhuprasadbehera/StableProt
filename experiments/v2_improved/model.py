"""
V2 Improved Model — MLP with Dropout, BatchNorm, and LogitsOutput.

Key improvements over V1 Baseline:
  1. BatchNorm after each hidden layer (stabilizes training)
  2. Dropout for regularization (prevents overfitting to majority class)
  3. NO Sigmoid output — uses raw logits with BCEWithLogitsLoss
     (more numerically stable + allows pos_weight for class imbalance)

Architecture: Linear(1024→512) → BN → ReLU → Dropout(0.3)
            → Linear(512→256)  → BN → ReLU → Dropout(0.2)
            → Linear(256→1)    [raw logits]
"""

import torch
from torch import nn


class MLP_Improved(nn.Module):
    """Improved MLP with Dropout + BatchNorm. Outputs raw logits."""

    def __init__(self, input_size=1024, hidden_size_1=512, hidden_size_2=256,
                 dropout_1=0.3, dropout_2=0.2):
        super(MLP_Improved, self).__init__()
        self.input_size = input_size

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.BatchNorm1d(hidden_size_1),
            nn.ReLU(),
            nn.Dropout(dropout_1),

            nn.Linear(hidden_size_1, hidden_size_2),
            nn.BatchNorm1d(hidden_size_2),
            nn.ReLU(),
            nn.Dropout(dropout_2),

            nn.Linear(hidden_size_2, 1),
            # No Sigmoid! Raw logits for BCEWithLogitsLoss
        )

    def forward(self, x):
        return self.model(x)

    def predict_proba(self, x):
        """Get probability output by applying sigmoid to logits."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x.float()).squeeze()
            return torch.sigmoid(logits)
