"""
V4 Improved Regression — Hyperparameters and configuration.

Improvements over V3:
  - Huber loss (robust to noisy OGT labels)
  - Target normalization (z-score)
  - LR scheduler (ReduceLROnPlateau)
  - Gradient clipping
  - Mixup augmentation
  - Per-sample bin weights
  - Residual connection in model
"""

CONFIG = {
    # Architecture
    'input_size': 1024,           # ProtT5
    'hidden_size_1': 512,
    'hidden_size_2': 256,
    'dropout_1': 0.3,
    'dropout_2': 0.2,

    # Training
    'learning_rate': 1e-4,        # Lower than V3 (1e-3)
    'batch_size': 64,
    'num_epochs': 50,
    'early_stopping_patience': 10,
    'weight_decay': 1e-5,

    # Loss
    'loss_type': 'huber',         # V3 used MSE
    'huber_delta': 5.0,

    # Improvements over V3
    'target_normalization': True,
    'lr_scheduler_patience': 5,
    'lr_scheduler_factor': 0.5,
    'grad_clip_max_norm': 1.0,
    'mixup_alpha': 0.2,

    # Seeds for ensemble
    'seeds': [1, 2, 3, 4, 5],

    # Data
    'data_path': 'prepared_data_full.pt',
}
