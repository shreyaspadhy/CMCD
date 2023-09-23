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
                "MCD_CAIS_sn": 1e-5,
                "MCD_U_a-lp-sn": 1e-3,
                "UHA": 1e-3,
                "MCD_ULA_sn": 1e-5,
                "MCD_ULA": 1e-5},
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
    config.model = "lorenz"
    config.N = 5 # 5 for all except NICE
    config.nbridges = 8
    config.lfsteps = 1

    config.init_eta = 0.0
    config.init_eps = 1e-5
    config.pretrain_mfvi = True

    config.train_vi = True
    config.train_eps = True

    config.mfvi_iters = 150000
    config.mfvi_lr = 0.01
    config.iters = 150000 # 150000 for all except NICE
    config.lr = 0.0001
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

    # Funnel configs
    config.funnel_d = 10
    config.funnel_sig = 3
    config.funnel_clipy = 11
    

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