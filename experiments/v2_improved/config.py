"""
V2 Improved — Hyperparameters and configuration.

Key differences from V1:
  - Weighted loss (BCEWithLogitsLoss + pos_weight)
  - Balanced sampling (WeightedRandomSampler)
  - Dropout + BatchNorm regularization
"""

CONFIG = {
    # Architecture — larger model for V2
    'input_size': 1024,
    'hidden_size_1': 512,
    'hidden_size_2': 256,
    'dropout_1': 0.3,
    'dropout_2': 0.2,

    # Training
    'learning_rate': 1e-3,
    'batch_size': 64,
    'num_epochs': 50,
    'early_stopping_patience': 10,
    'weight_decay': 1e-5,      # Light L2 regularization

    # Loss — BCEWithLogitsLoss with automatic pos_weight
    'loss_type': 'bce_weighted',
    'pos_weight_mode': 'auto',  # auto-calculate from class distribution
    'pos_weight_cap': 50.0,     # Cap extreme pos_weight values

    # Sampling
    'balanced_sampling': True,   # Use WeightedRandomSampler

    # Temperature thresholds to train classifiers for (0-100 range)
    'thresholds': list(range(5, 100, 5)),

    # Seeds for ensemble
    'seeds': [1, 2, 3, 4, 5],

    # Data
    'data_path': 'prepared_data_full.pt',
}
