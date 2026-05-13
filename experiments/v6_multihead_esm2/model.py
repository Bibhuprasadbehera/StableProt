import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPool(nn.Module):
    def __init__(self, dim=2560):
        super().__init__()
        self.attn = nn.Linear(dim, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, dim)
        # if x is already (batch, dim) from mean pooling, we just return x
        if x.dim() == 2:
            return x
        weights = F.softmax(self.attn(x), dim=1) # (batch, seq_len, 1)
        return (x * weights).sum(dim=1) # (batch, dim)

class MultiHead_TmPredictor(nn.Module):
    def __init__(self, input_size=2560, hidden1=512, hidden2=256, dropout1=0.3, dropout2=0.2, use_attention=False):
        super(MultiHead_TmPredictor, self).__init__()
        
        # Optional Attention Pooling for sequence-level features
        self.use_attention = use_attention
        if self.use_attention:
            self.attention_pool = AttentionPool(input_size)
        
        # Shared backbone
        self.fc1 = nn.Linear(input_size, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.dropout1 = nn.Dropout(dropout1)
        
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.dropout2 = nn.Dropout(dropout2)
        
        # Residual projection if hidden dimensions differ
        self.residual_proj = nn.Linear(hidden1, hidden2) if hidden1 != hidden2 else nn.Identity()
        
        # Two prediction heads
        self.head_ogt = nn.Linear(hidden2, 1)  # Head A: OGT
        self.head_tm = nn.Linear(hidden2, 1)   # Head B: Tm

    def forward_backbone(self, x):
        if hasattr(self, 'use_attention') and self.use_attention and x.dim() == 3:
            x = self.attention_pool(x)
            
        # First layer
        x1 = self.fc1(x)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = self.dropout1(x1)
        
        # Second layer with residual connection
        x2 = self.fc2(x1)
        x2 = self.bn2(x2)
        x2 = F.relu(x2 + self.residual_proj(x1)) # Residual connection
        x2 = self.dropout2(x2)
        
        return x2

    def forward(self, x, head='tm'):
        features = self.forward_backbone(x)
        if head == 'ogt':
            return self.head_ogt(features).squeeze(-1)
        else:
            return self.head_tm(features).squeeze(-1)
