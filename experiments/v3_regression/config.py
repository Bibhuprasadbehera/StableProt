"""
V3 Regression — Hyperparameters and configuration.
"""

CONFIG = {
    # Architecture
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
    'weight_decay': 1e-5,

    # Loss
    'loss_type': 'mse',        # MSELoss for regression

    # Seeds for ensemble
    'seeds': [1, 2, 3, 4, 5],

    # Data
    'data_path': '../prepared_data_full.pt',  # Default to full dataset if available
}
