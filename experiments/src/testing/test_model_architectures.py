import unittest
import os
import sys
import torch
import torch.nn as nn

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Try importing the models
try:
    from experiments.v5_multihead.model import MultiHead_TmPredictor as V5_Model
except ImportError:
    V5_Model = None

try:
    from experiments.v6_multihead_esm2.model import MultiHead_TmPredictor as V6_Model
except ImportError:
    V6_Model = None

class TestModelArchitectures(unittest.TestCase):

    def test_v5_multihead_prott5(self):
        """Test V5 (ProtT5) instantiation and forward passes"""
        if V5_Model is None:
            self.skipTest("V5 model not found")
            
        model = V5_Model(input_size=1024, hidden1=512, hidden2=256)
        model.eval()
        
        batch_size = 4
        dummy_input = torch.randn(batch_size, 1024)
        
        # Test default forward (should return both or main)
        out_ogt = model(dummy_input, head='ogt')
        out_tm = model(dummy_input, head='tm')
        
        self.assertEqual(out_ogt.shape, (batch_size,), "OGT head output shape mismatch (expected 1D squeezed tensor)")
        self.assertEqual(out_tm.shape, (batch_size,), "Tm head output shape mismatch (expected 1D squeezed tensor)")
        
        # Ensure outputs are different (heads are independent)
        self.assertFalse(torch.allclose(out_ogt, out_tm), "OGT and Tm heads should produce different outputs")

    def test_v6_multihead_esm2(self):
        """Test V6 (ESM-2) instantiation and forward passes"""
        if V6_Model is None:
            self.skipTest("V6 model not found")
            
        # ESM-2 3B has 2560 dimensions
        model = V6_Model(input_size=2560, hidden1=512, hidden2=256)
        model.eval()
        
        batch_size = 4
        dummy_input = torch.randn(batch_size, 2560)
        
        out_ogt = model(dummy_input, head='ogt')
        out_tm = model(dummy_input, head='tm')
        
        self.assertEqual(out_ogt.shape, (batch_size,), "OGT head output shape mismatch")
        self.assertEqual(out_tm.shape, (batch_size,), "Tm head output shape mismatch")

    def test_shared_backbone_gradients(self):
        """Test that gradients flow through the shared backbone from both heads"""
        if V6_Model is None:
            self.skipTest("V6 model not found")
            
        model = V6_Model(input_size=2560)
        model.train()
        
        dummy_input = torch.randn(2, 2560)
        
        # 1. Forward OGT and backward
        model.zero_grad()
        out_ogt = model(dummy_input, head='ogt')
        loss_ogt = out_ogt.sum()
        loss_ogt.backward()
        
        # Check that backbone has gradients (use fc1 as it's part of shared backbone)
        has_grad_from_ogt = any(p.grad is not None and p.grad.sum().item() != 0 for p in model.fc1.parameters())
        self.assertTrue(has_grad_from_ogt, "Gradients did not flow from OGT head to shared backbone")
        
        # 2. Forward Tm and backward
        model.zero_grad()
        out_tm = model(dummy_input, head='tm')
        loss_tm = out_tm.sum()
        loss_tm.backward()
        
        has_grad_from_tm = any(p.grad is not None and p.grad.sum().item() != 0 for p in model.fc1.parameters())
        self.assertTrue(has_grad_from_tm, "Gradients did not flow from Tm head to shared backbone")

if __name__ == '__main__':
    unittest.main()
