import os
import sys
import torch
import torch.nn as nn
from transformers import EsmTokenizer, EsmModel

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
