"""
V1 Baseline Model — Standard MLP with no regularization.
Matches the original TemStaPro MLP_C2H2 architecture exactly.

Architecture: Linear(1024→512) → ReLU → Linear(512→256) → ReLU → Linear(256→1) → Sigmoid
Loss: BCELoss (unweighted)
"""

import torch
from torch import nn


class MLP_Baseline(nn.Module):
    """Original TemStaPro classifier — no dropout, no batch norm."""

    def __init__(self, input_size=1024, hidden_size_1=512, hidden_size_2=256):
        super(MLP_Baseline, self).__init__()
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2

        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size_1),
            nn.ReLU(),
            nn.Linear(self.hidden_size_1, self.hidden_size_2),
            nn.ReLU(),
            nn.Linear(self.hidden_size_2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

    def predict_proba(self, x):
        """Get probability output (same as forward since Sigmoid is built in)."""
        self.eval()
        with torch.no_grad():
            return self.forward(x.float()).squeeze()
