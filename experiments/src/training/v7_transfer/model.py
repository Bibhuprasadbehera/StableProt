import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Pre-activation residual block with LayerNorm and GELU."""
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-activation: x + dropout(linear(activation(norm(x))))
        return x + self.dropout(self.linear(self.act(self.norm(x))))


class StableProtV7(nn.Module):
    """
    Two-stage transfer learning model for protein thermostability prediction.
    
    Stage 1: Pre-train backbone + ogt_head on OGT data (943K samples).
    Stage 2: Load pre-trained backbone. Train tm_head on Tm data (43K samples).
             Optionally use OGT prediction as auxiliary feature.
    
    Architecture:
        Input (emb_dim) → backbone (emb_dim→512→512→256) → head (256→1)
    """
    def __init__(self, emb_dim=2560, use_ogt_feature=False, use_tm_feature=False,
                 hidden=512, bottleneck=256, dropout1=0.3, dropout2=0.2):
        super().__init__()
        self.emb_dim = emb_dim
        self.use_ogt_feature = use_ogt_feature
        self.use_tm_feature = use_tm_feature

        # Input projections
        if emb_dim == 1280:
            # SaProt mode: separate input projections because OGT embeddings are ESM-2 (2560)
            self.input_layer_tm = nn.Linear(1280, hidden)
            self.input_layer_ogt = nn.Linear(2560, hidden)
        else:
            # ESM-2 mode: single input projection
            self.input_layer = nn.Linear(emb_dim, hidden)

        # Shared backbone (pre-trained on OGT in Stage 1)
        self.backbone_rest = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout1),
            ResidualBlock(hidden, dropout=dropout2),
            nn.Linear(hidden, bottleneck),
            nn.LayerNorm(bottleneck),
            nn.GELU(),
            nn.Dropout(dropout2),
        )

        # OGT prediction head (used in Stage 1, frozen/discarded in Stage 2)
        self.ogt_head = nn.Linear(bottleneck, 1)

        # Tm prediction head (used in Stage 2)
        tm_input_dim = bottleneck + (1 if use_ogt_feature else 0) + (1 if use_tm_feature else 0)
        self.tm_head = nn.Sequential(
            nn.Linear(tm_input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout2),
            nn.Linear(128, 1),
        )

    def forward_backbone(self, x, stage='tm'):
        """Extract backbone features from PLM embeddings."""
        if self.emb_dim == 1280:
            if stage == 'ogt':
                h = self.input_layer_ogt(x)
            else:
                h = self.input_layer_tm(x)
        else:
            h = self.input_layer(x)
        return self.backbone_rest(h)

    def predict_ogt(self, x):
        """Stage 1: Predict OGT from embeddings."""
        features = self.forward_backbone(x, stage='ogt')
        return self.ogt_head(features).squeeze(-1)

    def predict_tm(self, x, ogt_pred=None, tm_feat=None):
        """Stage 2: Predict Tm from embeddings (+ optional OGT and TM features)."""
        features = self.forward_backbone(x, stage='tm')
        features_list = [features]
        if self.use_ogt_feature and ogt_pred is not None:
            features_list.append(ogt_pred.unsqueeze(-1))
        if self.use_tm_feature and tm_feat is not None:
            features_list.append(tm_feat.unsqueeze(-1))
        
        if len(features_list) > 1:
            features = torch.cat(features_list, dim=-1)
        return self.tm_head(features).squeeze(-1)

    def forward(self, x, stage='tm', ogt_pred=None, tm_feat=None):
        """Unified forward pass."""
        if stage == 'ogt':
            return self.predict_ogt(x)
        else:
            return self.predict_tm(x, ogt_pred=ogt_pred, tm_feat=tm_feat)

    def predict_with_uncertainty(self, x, ogt_pred=None, tm_feat=None, n_samples=30):
        """
        MC-Dropout uncertainty estimation.
        Runs multiple forward passes with dropout enabled to calculate mean and std.
        """
        was_training = self.training
        self.train()  # Activate dropout
        
        preds = []
        for _ in range(n_samples):
            with torch.no_grad():
                preds.append(self.predict_tm(x, ogt_pred=ogt_pred, tm_feat=tm_feat))
                
        if not was_training:
            self.eval()  # Restore original mode
            
        preds = torch.stack(preds)
        return preds.mean(dim=0), preds.std(dim=0)

    def freeze_backbone(self):
        """Freeze backbone parameters for Stage 2 initial training."""
        if hasattr(self, 'input_layer'):
            for param in self.input_layer.parameters():
                param.requires_grad = False
        if hasattr(self, 'input_layer_tm'):
            for param in self.input_layer_tm.parameters():
                param.requires_grad = False
        if hasattr(self, 'input_layer_ogt'):
            for param in self.input_layer_ogt.parameters():
                param.requires_grad = False
        for param in self.backbone_rest.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning in Stage 2."""
        if hasattr(self, 'input_layer'):
            for param in self.input_layer.parameters():
                param.requires_grad = True
        if hasattr(self, 'input_layer_tm'):
            for param in self.input_layer_tm.parameters():
                param.requires_grad = True
        if hasattr(self, 'input_layer_ogt'):
            for param in self.input_layer_ogt.parameters():
                param.requires_grad = True
        for param in self.backbone_rest.parameters():
            param.requires_grad = True

    def get_stage2_param_groups(self, backbone_lr=1e-5, head_lr=1e-4):
        """Return parameter groups with differential learning rates for Stage 2."""
        backbone_params = list(self.backbone_rest.parameters())
        if hasattr(self, 'input_layer'):
            backbone_params.extend(list(self.input_layer.parameters()))
        if hasattr(self, 'input_layer_tm'):
            backbone_params.extend(list(self.input_layer_tm.parameters()))
        if hasattr(self, 'input_layer_ogt'):
            backbone_params.extend(list(self.input_layer_ogt.parameters()))
            
        return [
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': self.tm_head.parameters(), 'lr': head_lr},
        ]
