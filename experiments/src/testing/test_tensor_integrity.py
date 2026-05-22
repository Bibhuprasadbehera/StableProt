import unittest
import os
import sys
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class TestTensorIntegrity(unittest.TestCase):
    
    def setUp(self):
        self.prepared_data_path = os.path.join(project_root, "new_data", "prepared_data_v2_leak_free.pt")
        self.fireprot_data_path = os.path.join(project_root, "experiments", "data_processing", "fireprot_holdout_prott5.pt")

    def test_tensor_health_nans(self):
        """Scans embedding tensors for NaN or Inf values"""
        if os.path.exists(self.prepared_data_path):
            data = torch.load(self.prepared_data_path, map_location='cpu', weights_only=False)
            if 'train_tm' in data and 'embeddings' in data['train_tm']:
                emb = data['train_tm']['embeddings']
                self.assertFalse(torch.isnan(emb).any(), "NaN values found in Train Tm embeddings")
                self.assertFalse(torch.isinf(emb).any(), "Inf values found in Train Tm embeddings")
                
        if os.path.exists(self.fireprot_data_path):
            data = torch.load(self.fireprot_data_path, map_location='cpu', weights_only=False)
            if 'embeddings_prott5' in data:
                emb = data['embeddings_prott5']
                self.assertFalse(torch.isnan(emb).any(), "NaN values found in FireProt ProtT5 embeddings")
                self.assertFalse(torch.isinf(emb).any(), "Inf values found in FireProt ProtT5 embeddings")

    def test_shape_alignment(self):
        """Asserts that the number of embeddings perfectly matches the number of labels"""
        if os.path.exists(self.prepared_data_path):
            data = torch.load(self.prepared_data_path, map_location='cpu', weights_only=False)
            if 'train_tm' in data:
                emb = data['train_tm']['embeddings']
                labels = data['train_tm']['labels']
                self.assertEqual(emb.shape[0], labels.shape[0], "Mismatch between number of Train Tm embeddings and labels")
                
        if os.path.exists(self.fireprot_data_path):
            data = torch.load(self.fireprot_data_path, map_location='cpu', weights_only=False)
            if 'embeddings_prott5' in data and 'temperatures' in data:
                emb = data['embeddings_prott5']
                labels = data['temperatures']
                self.assertEqual(emb.shape[0], len(labels), "Mismatch between number of FireProt embeddings and labels")

if __name__ == '__main__':
    unittest.main()
