"""
V1 Baseline — Hyperparameters and configuration.
"""

CONFIG = {
    # Architecture — matches original pre-trained TemStaPro models (v0)
    # Change to 512/256 to experiment with a larger network
    'input_size': 1024,
    'hidden_size_1': 256,
    'hidden_size_2': 128,

    # Training
    'learning_rate': 1e-3,
    'batch_size': 64,
    'num_epochs': 50,
    'early_stopping_patience': 10,
    'weight_decay': 0.0,       # No regularization in baseline

    # Loss
    'loss_type': 'bce',        # Standard BCELoss (unweighted)
    'balanced_sampling': False, # No balanced sampling in baseline

    # Temperature thresholds to train classifiers for
    'thresholds': [40, 45, 50, 55, 60, 65],

    # Seeds for ensemble
    'seeds': [1, 2, 3, 4, 5],

    # Data
    'data_path': '../prepared_data.pt',
}
