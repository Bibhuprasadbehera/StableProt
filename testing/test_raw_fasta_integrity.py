import unittest
import os
import sys
from Bio import SeqIO

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class TestRawFastaIntegrity(unittest.TestCase):
    
    def setUp(self):
        self.fastas = [
            os.path.join(project_root, "new_data", "ogt_training_leak_free.fasta"),
            os.path.join(project_root, "new_data", "prothermdb_validation_clean.fasta"),
            os.path.join(project_root, "new_data", "meltome_sequences_leak_free.fasta")
        ]
        self.invalid_chars = set('BJOUXZ')

    def test_amino_acid_validity(self):
        """Asserts that sequences only contain standard amino acids"""
        for fasta_path in self.fastas:
            if not os.path.exists(fasta_path):
                continue
                
            with open(fasta_path, "r") as f:
                for i, record in enumerate(SeqIO.parse(f, "fasta")):
                    seq = str(record.seq).upper()
                    invalid = [c for c in seq if c in self.invalid_chars]
                    self.assertEqual(len(invalid), 0, f"Invalid amino acid {invalid} found in {os.path.basename(fasta_path)} at record {i}")

    def test_length_constraints(self):
        """Asserts all sequences are within valid biological and memory limits (>20, <1500)"""
        for fasta_path in self.fastas:
            if not os.path.exists(fasta_path):
                continue
                
            with open(fasta_path, "r") as f:
                for i, record in enumerate(SeqIO.parse(f, "fasta")):
                    seq_len = len(record.seq)
                    self.assertTrue(seq_len >= 20, f"Sequence too short ({seq_len}) in {os.path.basename(fasta_path)} at record {i}")
                    self.assertTrue(seq_len <= 1500, f"Sequence too long ({seq_len}) in {os.path.basename(fasta_path)} at record {i} - OOM risk")

    def test_internal_duplication(self):
        """Asserts there are no duplicate sequence entries within the same dataset"""
        for fasta_path in self.fastas:
            if not os.path.exists(fasta_path):
                continue
                
            seen = set()
            with open(fasta_path, "r") as f:
                for i, record in enumerate(SeqIO.parse(f, "fasta")):
                    seq = str(record.seq).upper()
                    self.assertNotIn(seq, seen, f"Exact duplicate sequence found in {os.path.basename(fasta_path)} at record {i}")
                    seen.add(seq)

if __name__ == '__main__':
    unittest.main()
