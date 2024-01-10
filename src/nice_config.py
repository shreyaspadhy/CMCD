import os

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.alpha = 0.05  # 0.000001
    config.batch_size = 1000
    config.hidden_dim = 1000
    config.im_size = 14  # 14
    config.log_interval = 1000
    config.lr = 0.0001
    config.n_bits = 3  # 3
    config.num_epochs = 500
    config.save_interval = 1000
    config.seed = 42
    config.weight_decay = 0.000001

    cwd = os.getcwd()
    # Wandb Configs
    config.wandb = ml_collections.ConfigDict()
    config.wandb.log = True
    config.wandb.project = "cais"
    config.wandb.entity = "shreyaspadhy"
    config.wandb.code_dir = cwd
    config.wandb.name = ""
    config.wandb.log_artifact = True

    return config
