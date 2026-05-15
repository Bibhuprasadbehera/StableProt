import esm
import torch
import os

def main():
    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    model = model.eval()
    batch_converter = alphabet.get_batch_converter()
    
    # Use the first sequence from the training data
    seq = "MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGNFGADAQGAMNKALELFRKDIAAKYKELGYQG"
    data = [("id1", seq)]
    labels, strs, tokens = batch_converter(data)
    
    with torch.no_grad():
        results = model(tokens, repr_layers=list(range(37)))
        
    for l in range(37):
        # Mean pooling excluding padding/CLS
        emb = results["representations"][l][0, 1:len(seq)+1].mean(0)
        print(f"Layer {l}: std {emb.std().item():.4f}")

if __name__ == "__main__":
    main()
