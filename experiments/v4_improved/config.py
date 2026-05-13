CONFIG = {
    # Architecture
    'input_size': 1024,           # ProtT5
    'hidden_size_1': 512,
    'hidden_size_2': 256,
    'dropout_1': 0.3,
    'dropout_2': 0.2,

    # Training — proven values from V3 + minimal additions
    'learning_rate': 1e-3,        # Same as V3 (1e-4 was too slow)
    'batch_size': 64,
    'num_epochs': 50,
    'early_stopping_patience': 10,
    'weight_decay': 1e-4,         # Slightly stronger than V3 (1e-5)

    # Loss — MSE (Huber hurt on clean OGT data)
    'loss_type': 'mse',

    # Improvements that actually help
    'target_normalization': False, # Unnecessary for clean OGT labels
    'lr_scheduler': 'cosine',     # CosineAnnealingLR instead of ReduceLROnPlateau
    'grad_clip_max_norm': 1.0,
    'mixup_alpha': 0.0,           # Disabled (hurt regression quality)

    # Seeds for ensemble
    'seeds': [1, 2, 3, 4, 5],

    # Data
    'data_path': 'prepared_data_full.pt',
}

