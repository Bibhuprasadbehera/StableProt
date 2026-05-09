import torch

data = torch.load("new_data/prepared_data_v2.pt", weights_only=True)

for key in data:
    d = data[key]
    if "embeddings" in d and d["embeddings"].numel() > 0:
        nan_embs = torch.isnan(d["embeddings"]).sum().item()
        inf_embs = torch.isinf(d["embeddings"]).sum().item()
        print(f"{key} Embeddings - NaNs: {nan_embs}, Infs: {inf_embs}")
    if "labels" in d and d["labels"].numel() > 0:
        nan_labels = torch.isnan(d["labels"]).sum().item()
        inf_labels = torch.isinf(d["labels"]).sum().item()
        print(f"{key} Labels - NaNs: {nan_labels}, Infs: {inf_labels}")
