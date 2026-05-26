import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHead_SaProtPredictor(nn.Module):
    def __init__(self, hidden1=512, hidden2=256, dropout1=0.3, dropout2=0.2):
        super().__init__()
        # Dual input projection layers
        self.input_layer_tm = nn.Linear(1280, hidden1)
        self.input_layer_ogt = nn.Linear(2560, hidden1)
        
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.dropout1 = nn.Dropout(dropout1)
        
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.dropout2 = nn.Dropout(dropout2)
        
        self.residual_proj = nn.Linear(hidden1, hidden2) if hidden1 != hidden2 else nn.Identity()
        self.head_ogt = nn.Linear(hidden2, 1)
        self.head_tm = nn.Linear(hidden2, 1)
        
    def forward(self, x, head='tm'):
        if head == 'tm':
            x1 = self.input_layer_tm(x)
        else:
            x1 = self.input_layer_ogt(x)
            
        x1 = self.dropout1(F.relu(self.bn1(x1)))
        x2 = self.dropout2(F.relu(self.bn2(self.fc2(x1)) + self.residual_proj(x1)))
        
        if head == 'ogt':
            return self.head_ogt(x2).squeeze(-1)
        else:
            return self.head_tm(x2).squeeze(-1)
