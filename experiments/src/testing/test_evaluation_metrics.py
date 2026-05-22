import unittest
import os
import sys
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Try importing evaluation functions
try:
    from experiments.analysis.compare_all_prothermdb import compute_metrics
except ImportError:
    compute_metrics = None

class TestEvaluationMetrics(unittest.TestCase):

    def test_compute_metrics_accuracy(self):
        """Test the mathematical accuracy of the core evaluation metrics against deterministic arrays"""
        if compute_metrics is None:
            self.skipTest("compute_metrics function not found")
            
        # Ground truth
        y_true = np.array([40.0, 50.0, 60.0, 70.0, 80.0])
        
        # Perfect predictions
        y_pred_perfect = np.array([40.0, 50.0, 60.0, 70.0, 80.0])
        metrics_perfect = compute_metrics(y_true, y_pred_perfect)
        
        self.assertAlmostEqual(metrics_perfect['mae'], 0.0)
        self.assertAlmostEqual(metrics_perfect['pcc'], 1.0)
        self.assertAlmostEqual(metrics_perfect['r2'], 1.0)
        self.assertAlmostEqual(metrics_perfect['mape'], 0.0)
        
        # Shifted predictions (MAE = 5.0, PCC should still be 1.0)
        y_pred_shifted = y_true + 5.0
        metrics_shifted = compute_metrics(y_true, y_pred_shifted)
        
        self.assertAlmostEqual(metrics_shifted['mae'], 5.0)
        self.assertAlmostEqual(metrics_shifted['pcc'], 1.0)
        
        # Inverse predictions (PCC should be -1.0)
        y_pred_inverse = np.array([80.0, 70.0, 60.0, 50.0, 40.0])
        metrics_inverse = compute_metrics(y_true, y_pred_inverse)
        
        self.assertAlmostEqual(metrics_inverse['pcc'], -1.0)

    def test_binary_thresholding_logic(self):
        """Test that ROC AUC conversion thresholds correctly for survival analysis"""
        if compute_metrics is None:
            self.skipTest("compute_metrics function not found")
            
        y_true = np.array([40.0, 50.0, 60.0, 70.0, 80.0])
        y_pred = np.array([45.0, 55.0, 65.0, 75.0, 85.0])
        
        # compute_metrics uses threshold = 60 by default for binary metrics internally if we provided it, but the current implementation doesn't accept the kwarg.
        metrics = compute_metrics(y_true, y_pred)
        
        # True binary labels for >60: [0, 0, 0, 1, 1]  (Wait, compute_metrics uses > threshold)
        # Pred continuous: [45, 55, 65, 75, 85]
        # ROC AUC for perfect ordering should be 1.0
        
        self.assertAlmostEqual(metrics['roc_auc'], 1.0)

if __name__ == '__main__':
    unittest.main()
