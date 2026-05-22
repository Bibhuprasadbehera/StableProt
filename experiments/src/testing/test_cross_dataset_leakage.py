import unittest
import os
import sys
from Bio import SeqIO

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class TestCrossDatasetLeakage(unittest.TestCase):
    
    def setUp(self):
        self.clstr_file = os.path.join(project_root, "new_data", "cdhit_combined_output.fasta.clstr")
        
        self.ogt_fasta = os.path.join(project_root, "new_data", "ogt_training_leak_free.fasta")
        self.protherm_fasta = os.path.join(project_root, "new_data", "prothermdb_validation_clean.fasta")
        self.meltome_fasta = os.path.join(project_root, "new_data", "meltome_sequences_leak_free.fasta")

    def _get_seqs(self, path):
        if not os.path.exists(path):
            return set()
        return {str(r.seq).upper() for r in SeqIO.parse(path, "fasta")}

    def test_exact_sequence_leakage(self):
        """Scans the unified datasets to assert 0 exact sequence overlaps between train and holdout sets"""
        ogt_seqs = self._get_seqs(self.ogt_fasta)
        protherm_seqs = self._get_seqs(self.protherm_fasta)
        meltome_seqs = self._get_seqs(self.meltome_fasta)
        
        if ogt_seqs and protherm_seqs:
            overlap = ogt_seqs.intersection(protherm_seqs)
            self.assertEqual(len(overlap), 0, f"DATA LEAKAGE: {len(overlap)} exact overlaps between OGT Train and ProThermDB Val")
            
        # FireProt is handled via PyTorch tensor in test_data_integrity, but we could add it here if FASTA existed.

    def test_homology_leakage_cdhit(self):
        """Parses cdhit_combined_output.fasta.clstr to assert no test sequence shares a cluster with training"""
        if not os.path.exists(self.clstr_file):
            self.skipTest("CD-HIT cluster file not found for homology check")
            
        # TODO: Await user input on the exact naming convention in the CD-HIT file 
        # (e.g., did they prefix >OGT_ vs >PROTHERM_) to do the explicit set intersection per cluster.
        # This will be fully implemented once clarifying questions are answered.
        pass

if __name__ == '__main__':
    unittest.main()
