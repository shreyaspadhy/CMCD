import numpyro
import numpyro.distributions as dist
import jax.numpy as np
import numpy as onp
import jax
from jax.flatten_util import ravel_pytree
import argparse
import boundingmachine as bm
import mcdboundingmachine as mcdbm
import opt
from model_handler import load_model
import pickle
import ml_collections.config_flags
import wandb
from absl import app, flags
from utils import flatten_nested_dict, update_config_dict, setup_training


ml_collections.config_flags.DEFINE_config_file(
    "config",
    "configs/base.py",
    "Training configuration.",
    lock_config=False,
)
FLAGS = flags.FLAGS


# Boundmodes
# 	- ULA uses MCD_ULA
# 	- MCD uses MCD_ULA_sn
#	- UHA uses UHA
# 	- LDVI uses MCD_U_a-lp-sn
#   - CAIS uses MCD_CAIS_sn

def main(config):
	wandb_kwargs = {
			"project": config.wandb.project,
			"entity": config.wandb.entity,
			"config": flatten_nested_dict(config.to_dict()),
			"name": config.wandb.name if config.wandb.name else None,
			"mode": "online" if config.wandb.log else "disabled",
			"settings": wandb.Settings(code_dir=config.wandb.code_dir),
		}
	with wandb.init(**wandb_kwargs) as run:
		setup_training(run)
		update_config_dict(config, run, {})

		print(config)
		iters_base=config.iters
		log_prob_model, dim = load_model(config.model)
		rng_key_gen = jax.random.PRNGKey(config.seed)

		# Train initial variational distribution to maximize the ELBO
		trainable=('vd',)
		params_flat, unflatten, params_fixed = bm.initialize(dim=dim, nbridges=0, trainable=trainable)
		grad_and_loss = jax.jit(jax.grad(bm.compute_bound, 1, has_aux = True), static_argnums = (2, 3, 4))
		losses, diverged, params_flat, tracker = opt.run(
			config, 0.01, iters_base, params_flat, unflatten, params_fixed,
			log_prob_model, grad_and_loss, trainable, rng_key_gen, log_prefix='pretrain')
		vdparams_init = unflatten(params_flat)[0]['vd']

		elbo_init = -np.mean(np.array(losses[-500:]))
		print('Done training initial parameters, got ELBO %.2f.' % elbo_init)
		wandb.log({'elbo_init': onp.array(elbo_init)})

		if config.boundmode == 'UHA':
			trainable = ('vd', 'eps', 'eta', 'mgridref_y')
			params_flat, unflatten, params_fixed = bm.initialize(dim=dim, nbridges=config.nbridges, eta=0.0, eps = 0.00001,
				lfsteps=config.lfsteps, vdparams=vdparams_init, trainable=trainable)
			grad_and_loss = jax.jit(jax.grad(bm.compute_bound, 1, has_aux = True), static_argnums = (2, 3, 4))

		elif 'MCD' in config.boundmode:
			trainable = ('vd', 'eps', 'eta', 'gamma', 'mgridref_y')
			params_flat, unflatten, params_fixed = mcdbm.initialize(dim=dim, nbridges=config.nbridges, vdparams=vdparams_init, eta=0.0, eps = 0.00001,
				trainable=trainable, mode=config.boundmode)
			grad_and_loss = jax.jit(jax.grad(mcdbm.compute_bound, 1, has_aux = True), static_argnums = (2, 3, 4))

		else:
			raise NotImplementedError('Mode %s not implemented.' % config.boundmode)

		losses, diverged, params_flat, tracker = opt.run(config, config.lr, config.iters, params_flat, unflatten, params_fixed, log_prob_model, grad_and_loss,
			trainable, rng_key_gen, log_prefix='train')

		final_elbo = -np.mean(np.array(losses[-500:]))
		print('Done training, got ELBO %.2f.' % final_elbo)
		wandb.log({'elbo_final': onp.array(final_elbo)})

		tracker['elbo_init'] = elbo_init
		tracker['elbo_final'] = final_elbo
		tracker["diverged"] = diverged

		print(elbo_init, final_elbo)

		params_train, params_notrain = unflatten(params_flat)
		params = {**params_train, **params_notrain}


if __name__ == "__main__":
    import os
    import sys

    # if sys.argv:
    #     # pass wandb API as argv[1] and set environment variable
    #     # 'python mll_optim.py MY_API_KEY'
        # os.environ["WANDB_API_KEY"] = sys.argv[1]

    # Adds jax flags to the program.
    jax.config.config_with_absl()

    # Snippet to pass configs into the main function.
    def _main(argv):
        del argv
        config = FLAGS.config
        main(config)

    app.run(_main)