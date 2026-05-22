import unittest
import os
import sys
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class TestLabelIntegrity(unittest.TestCase):
    
    def setUp(self):
        self.csvs = [
            (os.path.join(project_root, "new_data", "meltome_sequences_with_ogt.csv"), "OGT"),
            (os.path.join(project_root, "new_data", "prothermdb_validation_with_ogt.csv"), "Tm_(C)")
        ]
        self.min_val = 0
        self.max_val = 110

    def test_biological_bounds(self):
        """Asserts that all targets fall within strict plausible ranges"""
        for csv_path, target_col in self.csvs:
            if not os.path.exists(csv_path):
                continue
                
            df = pd.read_csv(csv_path)
            if target_col not in df.columns:
                self.fail(f"Target column '{target_col}' missing in {os.path.basename(csv_path)}")
                
            # Drop NaNs for the bounds check (NaNs checked in another test)
            valid_targets = df[target_col].dropna()
            
            out_of_bounds = valid_targets[(valid_targets < self.min_val) | (valid_targets > self.max_val)]
            self.assertEqual(len(out_of_bounds), 0, f"Found {len(out_of_bounds)} targets out of bounds in {os.path.basename(csv_path)}")

    def test_missing_values(self):
        """Asserts no NaN or null values exist in critical target columns"""
        for csv_path, target_col in self.csvs:
            if not os.path.exists(csv_path):
                continue
                
            df = pd.read_csv(csv_path)
            if target_col not in df.columns:
                self.fail(f"Target column '{target_col}' missing in {os.path.basename(csv_path)}")
                
            missing = df[target_col].isna().sum()
            self.assertEqual(missing, 0, f"Found {missing} NaN values in column '{target_col}' of {os.path.basename(csv_path)}")

if __name__ == '__main__':
    unittest.main()
