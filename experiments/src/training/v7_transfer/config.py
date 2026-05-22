CONFIG = {
    'input_size': 2560,           # ESM-2 3B
    'hidden_size': 512,
    'bottleneck_size': 256,
    'dropout_1': 0.3,
    'dropout_2': 0.2,
    'batch_size': 64,
    'seeds': [1, 2, 3],
    
    # Stage 1 (OGT pre-training)
    'stage1_epochs': 5,
    'stage1_lr': 1e-4,
    
    # Stage 2 (Tm fine-tuning)
    'stage2_epochs': 20,
    'stage2_backbone_lr': 1e-5,
    'stage2_head_lr': 1e-4,
    
    'early_stopping_patience': 10,
    'weight_decay': 1e-5,
    'loss_type': 'huber',
    'huber_delta': 5.0,
    'lr_scheduler_patience': 5,
    'lr_scheduler_factor': 0.5,
    'grad_clip_max_norm': 1.0,
}
