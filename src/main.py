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
from jax import scipy as jscipy
from configs.base import LR_DICT


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
#   - CAIS_UHA uses MCD_CAIS_UHA_sn

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
		# Load in the correct LR from sweeps
		try:
			new_vals = {"lr": LR_DICT[config.model][config.boundmode]}
		except KeyError:
			new_vals = {}
			raise ValueError('LR not found for model %s and boundmode %s' % (config.model, config.boundmode))
		
		update_config_dict(config, run, new_vals)

		print(config)
		iters_base=config.iters
		log_prob_model, dim = load_model(config.model)
		rng_key_gen = jax.random.PRNGKey(config.seed)

		train_rng_key_gen, eval_rng_key_gen = jax.random.split(rng_key_gen)

		# Train initial variational distribution to maximize the ELBO
		trainable=('vd',)
		params_flat, unflatten, params_fixed = bm.initialize(dim=dim, nbridges=0, trainable=trainable)
		grad_and_loss = jax.jit(jax.grad(bm.compute_bound, 1, has_aux = True), static_argnums = (2, 3, 4))
		losses, diverged, params_flat, tracker = opt.run(
			config, 0.01, iters_base, params_flat, unflatten, params_fixed,
			log_prob_model, grad_and_loss, trainable, train_rng_key_gen, log_prefix='pretrain')
		vdparams_init = unflatten(params_flat)[0]['vd']

		elbo_init = -np.mean(np.array(losses[-500:]))
		print('Done training initial parameters, got ELBO %.2f.' % elbo_init)
		wandb.log({'elbo_init': onp.array(elbo_init)})

		if config.boundmode == 'UHA':
			trainable = ('vd', 'eps', 'eta', 'mgridref_y')
			params_flat, unflatten, params_fixed = bm.initialize(dim=dim, nbridges=config.nbridges, eta=0.0, eps = 0.00001,
				lfsteps=config.lfsteps, vdparams=vdparams_init, trainable=trainable)
			grad_and_loss = jax.jit(jax.grad(bm.compute_bound, 1, has_aux = True), static_argnums = (2, 3, 4))

			loss_fn = jax.jit(bm.compute_bound, static_argnums = (2, 3, 4))

		elif 'MCD' in config.boundmode:
			trainable = ('vd', 'eps', 'eta', 'gamma', 'mgridref_y')
			params_flat, unflatten, params_fixed = mcdbm.initialize(dim=dim, nbridges=config.nbridges, vdparams=vdparams_init, eta=0.0, eps = 0.00001,
				trainable=trainable, mode=config.boundmode)
			grad_and_loss = jax.jit(jax.grad(mcdbm.compute_bound, 1, has_aux = True), static_argnums = (2, 3, 4))

			loss_fn = jax.jit(mcdbm.compute_bound, static_argnums = (2, 3, 4))

		else:
			raise NotImplementedError('Mode %s not implemented.' % config.boundmode)

		losses, diverged, params_flat, tracker = opt.run(config, config.lr, config.iters, params_flat, unflatten, params_fixed, log_prob_model, grad_and_loss,
			trainable, train_rng_key_gen, log_prefix='train')

		# Average over 30 seeds, 500 samples each after training is done.
		n_samples = config.n_samples
		n_input_dist_seeds = config.n_input_dist_seeds

		eval_losses = opt.sample(
			config, n_samples, n_input_dist_seeds, params_flat, unflatten, params_fixed, log_prob_model, loss_fn,
			eval_rng_key_gen, log_prefix='eval')

		# (n_input_dist_seeds, n_samples)
		eval_losses = np.array(eval_losses)

		# Calculate mean and std of ELBOs over 30 seeds
		final_elbos = -np.mean(eval_losses, axis=1)
		final_elbo = np.mean(final_elbos)
		final_elbo_std = np.std(final_elbos)

		# Calculate mean and std of log Zs over 30 seeds
		ln_numsamp = np.log(n_samples)

		final_ln_Zs = jscipy.special.logsumexp(-np.array(eval_losses), axis=1)  - ln_numsamp

		final_ln_Z = np.mean(final_ln_Zs)
		final_ln_Z_std = np.std(final_ln_Zs)

		print('Done training, got ELBO %.2f.' % final_elbo)
		print('Done training, got ln Z %.2f.' % final_ln_Z)

		wandb.log({
			'elbo_final': onp.array(final_elbo),
			'final_ln_Z': onp.array(final_ln_Z),
			'elbo_final_std': onp.array(final_elbo_std),
			'final_ln_Z_std': onp.array(final_ln_Z_std)
			})

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