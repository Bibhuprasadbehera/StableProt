import unittest
import os
import sys
import torch
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class TestDataIntegrity(unittest.TestCase):
    
    def setUp(self):
        self.prepared_data_path = os.path.join(project_root, "new_data", "prepared_data_v2_leak_free.pt")
        self.fireprot_data_path = os.path.join(project_root, "experiments", "data_processing", "fireprot_holdout_prott5.pt")

    def test_prepared_data_v2_structure(self):
        """Test if the main training data tensor exists and has correct structure"""
        if not os.path.exists(self.prepared_data_path):
            self.skipTest(f"Data file not found: {self.prepared_data_path}")
            
        data = torch.load(self.prepared_data_path, map_location='cpu', weights_only=False)
        
        # Check required keys
        self.assertIn('train_tm', data)
        self.assertIn('embeddings', data['train_tm'])
        self.assertIn('labels', data['train_tm'])
        
        emb_shape = data['train_tm']['embeddings'].shape
        # Should be 2D tensor: [N, 2560] for ESM-2 or [N, 1024] for ProtT5
        self.assertEqual(len(emb_shape), 2, "Embeddings should be a 2D tensor")
        self.assertIn(emb_shape[1], [1024, 2560], "Embedding dimension must be 1024 (ProtT5) or 2560 (ESM-2)")
        
        # Check label bounds (Tm is roughly 0 to 120 C)
        labels = data['train_tm']['labels']
        self.assertTrue(torch.all(labels >= 0) and torch.all(labels <= 120), "Tm labels should be between 0 and 120 C")

    def test_fireprot_holdout_structure(self):
        """Test if the FireProt OOD holdout data is correctly formatted"""
        if not os.path.exists(self.fireprot_data_path):
            self.skipTest(f"FireProt data file not found: {self.fireprot_data_path}")
            
        data = torch.load(self.fireprot_data_path, map_location='cpu', weights_only=False)
        
        self.assertIn('embeddings_prott5', data)
        self.assertIn('embeddings_esm2', data)
        self.assertIn('temperatures', data)
        self.assertIn('sequences', data)
        
        # Check dimensions
        n_seqs = len(data['sequences'])
        self.assertEqual(data['embeddings_prott5'].shape, (n_seqs, 1024))
        self.assertEqual(data['embeddings_esm2'].shape, (n_seqs, 2560))
        self.assertEqual(len(data['temperatures']), n_seqs)

    def test_data_leakage(self):
        """Verify no sequence overlap between training and FireProt test set"""
        if not os.path.exists(self.prepared_data_path) or not os.path.exists(self.fireprot_data_path):
            self.skipTest("Required data files missing for leakage test")
            
        train_data = torch.load(self.prepared_data_path, map_location='cpu', weights_only=False)
        test_data = torch.load(self.fireprot_data_path, map_location='cpu', weights_only=False)
        
        # Some old v2 data dicts might not have 'sequences', so we handle gracefully
        if 'sequences' not in train_data.get('train_tm', {}):
            self.skipTest("Training data missing 'sequences' key for exact overlap check")
            
        train_seqs = set(train_data['train_tm']['sequences'])
        test_seqs = set(test_data['sequences'])
        
        overlap = train_seqs.intersection(test_seqs)
        self.assertEqual(len(overlap), 0, f"DATA LEAKAGE DETECTED: {len(overlap)} sequences overlap between train and FireProt OOD test.")

if __name__ == '__main__':
    unittest.main()
