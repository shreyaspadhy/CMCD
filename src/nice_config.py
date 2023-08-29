import ml_collections


def get_config():
    
  config = ml_collections.ConfigDict()

  config.alpha = 0.05 #0.000001
  config.batch_size = 1000
  config.hidden_dim = 1000
  config.im_size = 28 # 14
  config.log_interval = 100
  config.lr = 0.0001
  config.n_bits = 8 # 3
  config.num_epochs = 500
  config.save_interval = 1000
  config.seed = 42
  config.weight_decay = 0.000001

  return config