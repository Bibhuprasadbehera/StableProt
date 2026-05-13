"""
V4 Improved Regression Model.

Same architecture as V3 but with residual connection around second hidden layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_Regression_Improved(nn.Module):
    """
    V4 Improved Regression MLP: Predicts continuous OGT temperature.
    Improvements over V3 MLP_Regression:
      - Residual connection around second hidden layer
      - Separate layers (not nn.Sequential) for flexibility
    """
    def __init__(self, input_size=1024, hidden_size_1=512, hidden_size_2=256,
                 dropout_1=0.3, dropout_2=0.2):
        super(MLP_Regression_Improved, self).__init__()

        # First hidden layer
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.bn1 = nn.BatchNorm1d(hidden_size_1)
        self.dropout1 = nn.Dropout(dropout_1)

        # Second hidden layer
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.bn2 = nn.BatchNorm1d(hidden_size_2)
        self.dropout2 = nn.Dropout(dropout_2)

        # Residual projection (hidden1 → hidden2 dims)
        self.residual_proj = nn.Linear(hidden_size_1, hidden_size_2) if hidden_size_1 != hidden_size_2 else nn.Identity()

        # Output head
        self.head = nn.Linear(hidden_size_2, 1)

    def forward(self, x):
        # First layer
        x1 = self.fc1(x)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = self.dropout1(x1)

        # Second layer with residual
        x2 = self.fc2(x1)
        x2 = self.bn2(x2)
        x2 = F.relu(x2 + self.residual_proj(x1))
        x2 = self.dropout2(x2)

        return self.head(x2)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            return output.squeeze(-1)
