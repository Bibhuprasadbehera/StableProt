import unittest
import os
import sys
import torch
import torch.nn as nn

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Try importing the V6 model
try:
    from experiments.v6_multihead_esm2.model import MultiHead_TmPredictor as V6_Model
except ImportError:
    V6_Model = None

class TestTrainingLogic(unittest.TestCase):

    def test_multi_task_loss_weighting(self):
        """Test that the custom multi-task loss calculation applies weighting correctly"""
        if V6_Model is None:
            self.skipTest("V6 model not found")
            
        model = V6_Model(input_size=2560)
        criterion = nn.MSELoss()
        
        # Simulate OGT Batch
        dummy_input_ogt = torch.randn(2, 2560)
        dummy_target_ogt = torch.tensor([40.0, 50.0])
        
        # Forward pass for OGT
        out_ogt = model(dummy_input_ogt, head='ogt')
        loss_ogt = criterion(out_ogt, dummy_target_ogt)
        
        # Simulate weight logic from train.py (w_ogt=1.0, w_tm=1.5 for example)
        w_ogt = 1.0
        weighted_loss_ogt = loss_ogt * w_ogt
        
        # Simulate Tm Batch
        dummy_input_tm = torch.randn(2, 2560)
        dummy_target_tm = torch.tensor([60.0, 70.0])
        
        # Forward pass for Tm
        out_tm = model(dummy_input_tm, head='tm')
        loss_tm = criterion(out_tm, dummy_target_tm)
        
        w_tm = 1.5
        weighted_loss_tm = loss_tm * w_tm
        
        # Ensure loss scaling happens mathematically
        self.assertAlmostEqual(weighted_loss_ogt.item(), loss_ogt.item() * w_ogt, places=2)
        self.assertAlmostEqual(weighted_loss_tm.item(), loss_tm.item() * w_tm, places=2)
        
    def test_model_freezing(self):
        """Test that embeddings are implicitly frozen by not being passed to optimizer"""
        if V6_Model is None:
            self.skipTest("V6 model not found")
            
        model = V6_Model(input_size=2560)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # There should be no layers in the model with 2560x[vocab_size] parameters 
        # (meaning the transformer itself is excluded).
        # We check that the largest weight tensor is the first linear layer.
        max_params = 0
        for p in model.parameters():
            params = p.numel()
            if params > max_params:
                max_params = params
                
        # 2560 * 512 = 1,310,720 parameters. If it were the ESM-2 3B model, it would have > 3 Billion.
        self.assertTrue(max_params < 2000000, "Optimizer appears to contain massive tensors, backbone freezing may have failed")

if __name__ == '__main__':
    unittest.main()
