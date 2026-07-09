"""Configuration file for StableProt V8 Disjoint Multi-Head Architecture."""

CONFIG = {
    # Raw feature dimensions prior to bottleneck projection:
    # 1289 = 1280 SaProt embedding + 9 raw auxiliary features for Tm (OGT prior, TM flag, length, 6 AA ratios)
    # 1288 = 1280 SaProt embedding + 8 raw auxiliary features for OGT (TM flag, length, 6 AA ratios)
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
    'early_stopping_patience': 15,
    'huber_delta_tm': 5.0,
    'huber_delta_ogt': 15.0,
    'grad_clip_max_norm': 1.0,
    'seeds': [1, 2, 3, 4, 5],
    'bin_edges': list(range(0, 106, 5)),
    'weight_clamp_min': 0.3,
    'weight_clamp_max': 22.0,
    'weight_power': 0.75,
    'target_jitter_std': 0.5,
    'tm_ogt_noise_std': 6.0,
    # Auxiliary bottleneck projection dimension (Linear(9, 64) and Linear(8, 64)).
    # Concatenated dimension entering the first MLP hidden layer fc1 is 1280 + 64 = 1344 dims.
    'proj_dim': 64,
    'augment_noise_std': 0.02,
    'augment_prob': 0.15,
    'use_tta': False,
    # V8 specific hyperparameters
    'ogt_normalize': True,
    'ogt_subsample_meso_rate': 0.14,
    'iqr_scale': 6.34,
    'iqr_impute_val': 0.62,
    'focal_gamma': 2.0,
    'focal_beta': 0.5,
    'mixup_alpha': 0.2,
    'scheduler_type': 'cosine',
    'scheduler_T0': 10,
    'scheduler_Tmult': 2,
}
