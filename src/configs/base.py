import ml_collections
import os

def get_config():
    config = ml_collections.ConfigDict()
    config.boundmode = "UHA"
    config.model = "log_sonar"
    config.N = 5
    config.nbridges = 8
    config.lfsteps = 1
    config.iters = 15000
    config.lr = 0.001
    config.seed = 1
    config.id = -1
    config.run_cluster = 0

    cwd = os.getcwd()
    config.savedir = os.path.join(cwd, "results", config.model, config.boundmode, str(config.id))

    # Wandb Configs
    config.wandb = ml_collections.ConfigDict()
    config.wandb.log = True
    config.wandb.project = "cais"
    config.wandb.entity = "shreyaspadhy"
    config.wandb.code_dir = cwd
    config.wandb.name = ""
    config.wandb.log_artifact = False

    return config