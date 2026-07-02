"""Configuration file for StableProt V8 Disjoint Multi-Head Architecture."""

CONFIG = {
    'input_size_tm': 1289,
    'input_size_ogt': 1288,
    'hidden_size_1': 512,
    'hidden_size_2': 256,
    'dropout_1': 0.3,
    'dropout_2': 0.2,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'batch_size': 64,
    'max_epochs': 100,
    'early_stopping_patience': 10,
    'huber_delta': 5.0,
    'grad_clip_max_norm': 1.0,
    'seeds': [1, 2, 3, 4, 5],
    'bin_edges': list(range(0, 106, 5)),
    'weight_clamp_min': 0.5,
    'weight_clamp_max': 5.0,
    'target_jitter_std': 0.3,
}
