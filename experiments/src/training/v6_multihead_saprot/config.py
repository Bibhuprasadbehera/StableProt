CONFIG = {
    'input_size': 1280,           # SaProt
    'hidden_size_1': 512,
    'hidden_size_2': 256,
    'dropout_1': 0.3,
    'dropout_2': 0.2,
    'learning_rate': 1e-4,
    'batch_size': 64,
    'num_epochs': 50,
    'early_stopping_patience': 10,
    'weight_decay': 1e-5,
    'loss_type': 'huber',
    'huber_delta': 5.0,
    'ogt_loss_weight': 0.3,       # OGT head loss weight
    'tm_loss_weight': 1.0,        # Tm head loss weight
    'lr_scheduler_patience': 5,
    'lr_scheduler_factor': 0.5,
    'grad_clip_max_norm': 1.0,
    'mixup_alpha': 0.0,            # Disabled: causes target scale shifting in multi-head
    'target_normalization': False,  # Disabled: different z-scores for OGT/Tm heads cause gradient conflict
    'seeds': [1, 2, 3, 4, 5],
}
