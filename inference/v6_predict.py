import os
import torch
import torch.nn as nn
from transformers import EsmTokenizer, EsmModel

class MultiHeadSaProtV6Plus(nn.Module):
    def __init__(self, hidden1=512, hidden2=256, dropout1=0.3, dropout2=0.2):
        super().__init__()
        # TM pathway
        self.input_layer_tm = nn.Linear(1280, hidden1)
        self.bn1_tm = nn.BatchNorm1d(hidden1)
        self.fc2_tm = nn.Linear(hidden1, hidden2)
        self.bn2_tm = nn.BatchNorm1d(hidden2)
        self.residual_proj_tm = nn.Linear(hidden1, hidden2)
        self.head_tm = nn.Linear(hidden2, 1)
        
        # OGT pathway  
        self.input_layer_ogt = nn.Linear(2560, hidden1)
        self.bn1_ogt = nn.BatchNorm1d(hidden1)
        self.fc2_ogt = nn.Linear(hidden1, hidden2)
        self.bn2_ogt = nn.BatchNorm1d(hidden2)
        self.residual_proj_ogt = nn.Linear(hidden1, hidden2)
        self.head_ogt = nn.Linear(hidden2, 1)
        
        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)
        
    def forward(self, x, head='tm'):
        if head == 'tm':
            x1 = self.input_layer_tm(x)
            x1 = self.dropout1(torch.relu(self.bn1_tm(x1)))
            x2 = self.dropout2(torch.relu(self.bn2_tm(self.fc2_tm(x1)) + self.residual_proj_tm(x1)))
            return self.head_tm(x2).squeeze(-1)
        else:
            x1 = self.input_layer_ogt(x)
            x1 = self.dropout1(torch.relu(self.bn1_ogt(x1)))
            x2 = self.dropout2(torch.relu(self.bn2_ogt(self.fc2_ogt(x1)) + self.residual_proj_ogt(x1)))
            return self.head_ogt(x2).squeeze(-1)

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

class V6Predictor:
    def __init__(self, model_weights_path: str, device="cuda"):
        self.device = device
        # Ensure model weights exist
        if not os.path.exists(model_weights_path):
            raise FileNotFoundError(f"Weights not found at {model_weights_path}")
            
        # Load SaProt embedding model
        # Using 650M as it was standard for v6, but can be configured
        self.embed_model, self.tokenizer = load_saprot_model(device=device)
        
        # Load V6 prediction head
        self.v6_head = MultiHeadSaProtV6Plus().to(device)
        self.v6_head.load_state_dict(torch.load(model_weights_path, map_location=device))
        self.v6_head.eval()

    def predict_tm(self, sequence: str) -> float:
        embedding = get_saprot_embedding(self.embed_model, self.tokenizer, sequence, device=self.device)
        with torch.no_grad():
            pred = self.v6_head(embedding, head='tm')
        return pred.item()
