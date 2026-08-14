"""Configuration file for StableProt V9 Disjoint Multi-Head Architecture."""

import os

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
    'learning_rate': 1e-4,            # Reduced from 1e-3 to prevent OOD overfitting
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
    # V8=6.0. The sweep that chose 2.0 was scored on validation with the *true* prior, where
    # less noise always looks better; under the deployment prior it is not. Set to 0 when
    # tm_prior_source='predicted', since that prior already carries its own error.
    'tm_ogt_noise_std': float(os.environ.get('SP_TM_OGT_NOISE', '2.0')),
    # Auxiliary bottleneck projection dimension (Linear(9, 64) and Linear(8, 64)).
    # Concatenated dimension entering the first MLP hidden layer fc1 is 1280 + 64 = 1344 dims.
    'proj_dim': 64,
    'augment_noise_std': 0.02,
    'augment_prob': 0.15,
    'use_tta': False,
    # V9 specific hyperparameters
    'ogt_normalize': True,
    'ogt_subsample_meso_rate': 0.14,  # Keep 0.14 (strictly optimal balance from tests).
    'iqr_scale': 6.34,
    'iqr_impute_val': 0.62,
    'focal_gamma': 2.0,
    'focal_beta': 0.5,
    'mixup_alpha': 0.0,               # Set to 0.0 (disabled, was broken/not implemented).
    'scheduler_type': 'cosine',
    'scheduler_T0': 10,
    'scheduler_Tmult': 2,
    'use_residuals': True,            # Enabled residuals for better OOD regularization
    # When True the OGT head emits (mean, variance) and trains under Gaussian NLL, matching the Tm
    # head. When False it emits a point estimate under focal Huber and its only uncertainty is
    # ensemble seed spread, which underestimates predictive error by roughly 4x.
    'ogt_heteroscedastic': True,
    # 'huber_nll' trains the mean with the unchanged v9 point loss and the variance against a
    # detached mean, so OGT accuracy is held fixed while a real sigma is added. 'nll' swaps the
    # point loss for pure Gaussian NLL and will move the point predictions.
    'ogt_loss_mode': 'huber_nll',
    'ogt_var_loss_weight': 1.0,
    # Tm variance objective. Under 'nll' (the v9 behaviour) the mean and the variance are trained
    # by one joint Gaussian NLL, and the variance collapses onto a function of the predicted mean:
    # on ProThermDB sigma correlates with predicted Tm at rho = 0.58 but with realised error at
    # rho = 0.01, so a single constant width scores better. 'huber_nll' trains the mean with focal
    # Huber and the variance against a detached mean, the treatment that gave the OGT head a
    # calibrated sigma.
    'tm_loss_mode': 'huber_nll',
    'tm_var_loss_weight': 1.0,
    # --- OGT -> Tm coupling ---------------------------------------------------------------
    # The two pathways share no trainable parameters, so aux[0] (the OGT prior) is the only
    # thing that makes this one model rather than two. A constant prior is a constant column
    # into a linear layer, absorbed into its bias, i.e. exactly zero coupling. These options
    # keep the link and instead make it robust.
    #
    #   tm_prior_source   'true'      the v9 behaviour: train on the true OGT, serve a
    #                                 prediction that is off by ~17 C. A train/serve mismatch.
    #                     'predicted' train on the frozen OGT ensemble's own prediction, so the
    #                                 head sees at training time the prior it gets at inference.
    #   tm_use_ogt_sigma  also feed the prior's uncertainty, letting the Tm head discount an
    #                     unreliable prior per protein. Justified by sigma_OGT ranking prior
    #                     error at Spearman 0.41 on Tm proteins. Widens aux 9 -> 10.
    'tm_prior_source': os.environ.get('SP_TM_PRIOR_SOURCE', 'true'),
    'tm_use_ogt_sigma': os.environ.get('SP_TM_USE_OGT_SIGMA', '0') == '1',
    'tm_prior_ckpt_dir': os.environ.get('SP_TM_PRIOR_CKPT', 'experiments/src/training/v10/results'),
    'seq_len_min': 50,                # V8=50. Sweep verified 150 is better in-distribution but hurts OOD. Retrain with 50.
}
