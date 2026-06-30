import os
import sys
import torch
import torch.nn as nn
from transformers import EsmTokenizer, EsmModel

# Add training dir to path so we can import MultiHeadSaProtV7 if needed or define it
class MultiHeadSaProtV7(nn.Module):
    def __init__(self, input_dim=1280, hidden1=512, hidden2=256, dropout1=0.3, dropout2=0.2):
        super().__init__()
        self.shared_layer1 = nn.Linear(input_dim, hidden1)
        self.shared_bn1 = nn.LayerNorm(hidden1)
        self.shared_layer2 = nn.Linear(hidden1, hidden2)
        self.shared_bn2 = nn.LayerNorm(hidden2)
        self.shared_residual = nn.Linear(hidden1, hidden2)

        self.head_tm = nn.Linear(hidden2, 1)
        self.head_ogt = nn.Linear(hidden2, 1)

        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)

    def forward(self, x, task='tm'):
        x1 = self.dropout1(torch.relu(self.shared_bn1(self.shared_layer1(x))))
        x2 = self.dropout2(torch.relu(
            self.shared_bn2(self.shared_layer2(x1)) + self.shared_residual(x1)
        ))
        if task == 'tm':
            return self.head_tm(x2).squeeze(-1)
        elif task == 'ogt':
            return self.head_ogt(x2).squeeze(-1)
        else:
            raise ValueError(f"Unknown task: {task}")

def load_saprot_model(model_name="westlake-repl/SaProt_650M_AF2", device="cuda"):
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).to(device)
    model.eval()
    return model, tokenizer

def mask_sequence_for_saprot(seq: str) -> str:
    return "".join(f"{aa}#" for aa in seq)

def get_saprot_embedding(model, tokenizer, sequence: str, device="cuda") -> torch.Tensor:
    saprot_seq = mask_sequence_for_saprot(sequence)
    inputs = tokenizer(
        [saprot_seq],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(device)
    
    with torch.no_grad(), torch.amp.autocast('cuda' if 'cuda' in device else 'cpu'):
        outputs = model(**inputs)
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        hidden = outputs.last_hidden_state
        masked_hidden = hidden * attention_mask
        embeddings = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)
        
    return embeddings

class V7Predictor:
    def __init__(self, models_dir: str, device="cuda"):
        self.device = device
        self.embed_model, self.tokenizer = load_saprot_model(device=device)
        self.models = []
        for s in range(1, 6):
            p = os.path.join(models_dir, f"seed{s}/best_model.pt")
            if os.path.exists(p):
                m = MultiHeadSaProtV7(input_dim=1280).to(device)
                m.load_state_dict(torch.load(p, map_location=device, weights_only=False))
                m.eval()
                self.models.append(m)
        if not self.models:
            raise FileNotFoundError(f"No V7 seed models found in {models_dir}")

    def predict(self, sequence: str):
        emb = get_saprot_embedding(self.embed_model, self.tokenizer, sequence, device=self.device)
        emb = emb.float()
        tm_preds = []
        ogt_preds = []
        with torch.no_grad():
            for m in self.models:
                tm_preds.append(m(emb, task='tm').item())
                ogt_preds.append(m(emb, task='ogt').item())
        tm_val = sum(tm_preds) / len(tm_preds)
        ogt_val = sum(ogt_preds) / len(ogt_preds)
        return tm_val, ogt_val
