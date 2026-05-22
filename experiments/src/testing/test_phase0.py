import os
import sys
import time
import hashlib
import torch

def test_environment():
    print("="*60)
    print("  PHASE 0: ENVIRONMENT & CUDA VERIFICATION (Step 0.1)")
    print("="*60)
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    
    if not torch.cuda.is_available():
        print("\n[CRITICAL ERROR]: CUDA is NOT available! ESM-2 3B requires a GPU.")
        return False
        
    print("CUDA available: True")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    
    # Check VRAM
    free_mem, total_mem = torch.cuda.mem_get_info()
    free_gb = free_mem / (1024**3)
    total_gb = total_mem / (1024**3)
    print(f"GPU VRAM: {free_gb:.1f}GB free / {total_gb:.1f}GB total")
    
    if total_gb < 15:
        print("\n[WARNING]: ESM-2 3B might Out-Of-Memory (OOM) on GPUs with <15GB VRAM.")
    
    try:
        import esm
        print("\n[SUCCESS]: 'esm' library imported successfully.")
    except ImportError:
        print("\n[CRITICAL ERROR]: 'esm' library not installed. Run: pip install fair-esm")
        return False
        
    return True

def test_esm2_model(layer_to_extract=22):
    print("\n" + "="*60)
    print("  PHASE 0: ESM-2 3B LOADING & INFERENCE (Steps 0.2 & 0.4)")
    print("="*60)
    import esm
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading ESM-2 3B model (this may take a few minutes if downloading weights)...")
    start_time = time.time()
    try:
        # Load the massive 3B model
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        model = model.eval().to(device)
        print(f"[SUCCESS]: Model loaded in {time.time() - start_time:.1f}s.")
    except Exception as e:
        print(f"\n[CRITICAL ERROR]: Failed to load ESM-2 3B: {e}")
        return False
        
    batch_converter = alphabet.get_batch_converter()
    
    # Dummy sequences for test (one short, one slightly longer)
    data = [
        ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
        ("protein2", "MKNINPTQTAALERLTSLVKAQKEQAAEVLNAAEATIAEKEEDIKQVLEAAKAQALANELQEA")
    ]
    
    print(f"\nTesting inference on {len(data)} sequences...")
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    
    with torch.no_grad():
        try:
            # Extract specific layer (Layer 22 is the verified manifold for V6)
            results = model(batch_tokens, repr_layers=[layer_to_extract], return_contacts=False)
            token_representations = results["representations"][layer_to_extract]
            print(f"[SUCCESS]: Inference complete. Output tensor shape: {token_representations.shape}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n[CRITICAL ERROR]: OOM (Out of Memory) during inference! Try reducing batch size.\n{e}")
            else:
                print(f"\n[CRITICAL ERROR]: Inference error: {e}")
            return False
            
    print("\n" + "="*60)
    print("  PHASE 0: CACHING & HASHING LOGIC (Steps 0.3 & 0.5)")
    print("="*60)
    
    # Use a dummy test directory so we don't mess with real data
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "esm2_embeddings_cache_test")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Verified directory creation: {cache_dir}")
    
    for i, (_, seq) in enumerate(data):
        # 1. Pool (Mean of sequence excluding CLS [0] and EOS [seq_len+1] tokens)
        seq_len = len(seq)
        mean_rep = token_representations[i, 1 : seq_len + 1].mean(0).cpu()
        
        # 2. Hash (SHA256 of first 1500 chars to match original logic)
        seq_hash = hashlib.sha256(seq[:1500].encode()).hexdigest()
        out_path = os.path.join(cache_dir, f"esm2_{seq_hash}.pt")
        
        # 3. Save
        torch.save(mean_rep, out_path)
        print(f"[SUCCESS]: Cached {batch_labels[i]} (Length: {seq_len}) -> esm2_{seq_hash[:8]}...pt")
        
        # 4. Verify shape matches expected ESM-2 3B dim (2560)
        assert mean_rep.shape[0] == 2560, f"Shape mismatch! Expected 2560, got {mean_rep.shape[0]}"
        
    print("\n" + "🚀"*3 + " ALL PHASE 0 TESTS PASSED SUCESSFULLY! YOU ARE READY TO RUN THE FULL PIPELINE. " + "🚀"*3)
    return True

if __name__ == "__main__":
    if test_environment():
        test_esm2_model(layer_to_extract=22)
