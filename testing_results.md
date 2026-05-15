# StableProt V2: Exhaustive Data Validation Results

> **Note**: This report tracks structural, biological, and leakage validations for all core datasets.

## 1. Sequence Validations (FASTA)

### OGT Train (Cleaned)
- Total Sequences: 937258
- Min/Max Length: 22 - 1500
- ✅ Sequence Integrity: PASS

### ProThermDB Val (Cleaned)
- Total Sequences: 5221
- Min/Max Length: 27 - 1500
- ✅ Sequence Integrity: PASS

### Meltome Train (Cleaned)
- Total Sequences: 21319
- Min/Max Length: 20 - 1500
- ✅ Sequence Integrity: PASS

## 2. Label Validations (CSV)

### new_data/meltome_sequences_with_ogt.csv
- Total Rows: 25354
- ✅ Label Integrity: PASS

### new_data/prothermdb_validation_with_ogt.csv
- Total Rows: 5504
- ✅ Label Integrity: PASS

## 3. Cross-Dataset Leakage (Exact Match)

- **Train Tm vs FireProt OOD**: 0 exact overlaps detected.
- **OGT Train vs ProThermDB Val**: 0 exact overlaps detected.
