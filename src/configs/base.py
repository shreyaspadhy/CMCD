import ml_collections
import os


LR_DICT = {
    "log_sonar": {"MCD_CAIS_UHA_sn": 1e-3,
                  "MCD_CAIS_sn": 1e-3,
                  "MCD_U_a-lp-sn": 1e-3,
                  "UHA": 1e-4,
                  "MCD_ULA_sn": 1e-3,
                  "MCD_ULA": 1e-4},
    "log_ionosphere": {"MCD_CAIS_UHA_sn": 1e-3,
                       "MCD_CAIS_sn": 1e-4,
                       "MCD_U_a-lp-sn": 1e-3,
                       "UHA": 1e-4,
                       "MCD_ULA_sn": 1e-3,
                       "MCD_ULA": 1e-4},
    "lorenz": {"MCD_CAIS_UHA_sn": 1e-3,
                "MCD_CAIS_sn": 1e-4,
                "MCD_U_a-lp-sn": 1e-3,
                "UHA": 1e-3,
                "MCD_ULA_sn": 1e-4,
                "MCD_ULA": 1e-4},
    "brownian": {"MCD_CAIS_UHA_sn": 1e-3,
                "MCD_CAIS_sn": 1e-3,
                "MCD_U_a-lp-sn": 1e-3,
                "UHA": 1e-4,
                "MCD_ULA_sn": 1e-4,
                "MCD_ULA": 1e-5},
    "seeds": {"MCD_CAIS_UHA_sn": 1e-3,
            "MCD_CAIS_sn": 1e-3,
            "MCD_U_a-lp-sn": 1e-3,
            "UHA": 1e-3,
            "MCD_ULA_sn": 1e-3,
            "MCD_ULA": 1e-4},
    "banana": {"MCD_CAIS_UHA_sn": 1e-3,
            "MCD_CAIS_sn": 1e-3,
            "MCD_U_a-lp-sn": 1e-3,
            "UHA": 1e-3,
            "MCD_ULA_sn": 1e-3,
            "MCD_ULA": 1e-4},
}


def get_config():
    config = ml_collections.ConfigDict()
    config.boundmode = "UHA"
    config.model = "log_sonar"
    config.N = 5
    config.nbridges = 8
    config.lfsteps = 1

    config.mfvi_iters = 150000
    config.iters = 15000
    config.lr = 0.001
    config.seed = 1
    config.id = -1
    config.run_cluster = 0
    config.n_samples = 500
    config.n_input_dist_seeds = 30

    # NICE Config/
    config.im_size = 14
    config.alpha = 0.05
    config.n_bits = 3
    config.hidden_dim = 1000

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