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
from utils import flatten_nested_dict, update_config_dict, setup_training, make_grid, W2_distance
from jax import scipy as jscipy
from configs.base import LR_DICT


ml_collections.config_flags.DEFINE_config_file(
    "config",
    "configs/base.py",
    "Training configuration.",
    lock_config=False,
)
FLAGS = flags.FLAGS

# python main.py --config.model funnel --config.boundmode MCD_ULA

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
			if config.model == "nice":
				config.model = run.config.model + f"_{run.config.alpha}_{run.config.n_bits}_{run.config.im_size}"
				new_vals = {}
			elif config.model in ["funnel"]:
				new_vals = {}
			else:
				new_vals = {"lr": LR_DICT[run.config.model][run.config.boundmode]}
				print(new_vals)
		except KeyError:
			new_vals = {}
			raise ValueError('LR not found for model %s and boundmode %s' % (run.config.model, run.config.boundmode))
		# new_vals = {}
		update_config_dict(config, run, new_vals)

		print(config)

		if config.model in ['nice', 'funnel']:
			log_prob_model, dim, sample_from_target_fn = load_model(config.model, config)
		else:
			log_prob_model, dim = load_model(config.model, config)
			sample_from_target_fn = None
		rng_key_gen = jax.random.PRNGKey(config.seed)

		train_rng_key_gen, eval_rng_key_gen = jax.random.split(rng_key_gen)

		# Train initial variational distribution to maximize the ELBO
		trainable=('vd',)
		params_flat, unflatten, params_fixed = bm.initialize(dim=dim, nbridges=0, trainable=trainable)

		
		grad_and_loss = jax.jit(jax.grad(bm.compute_bound, 1, has_aux = True), static_argnums = (2, 3, 4))
		if not config.pretrain_mfvi:
			mfvi_iters = 1
			vdparams_init = unflatten(params_flat)[0]['vd']
		else:
			mfvi_iters = config.mfvi_iters
			losses, _, params_flat, _ = opt.run(
				config, config.mfvi_lr, mfvi_iters, params_flat, unflatten, params_fixed,
				log_prob_model, grad_and_loss, trainable, train_rng_key_gen, log_prefix='pretrain')
			vdparams_init = unflatten(params_flat)[0]['vd']

			elbo_init = -np.mean(np.array(losses[-500:]))
			print('Done training initial parameters, got ELBO %.2f.' % elbo_init)
			wandb.log({'elbo_init': onp.array(elbo_init)})

		if config.boundmode == 'UHA':
			trainable = ('eta', 'mgridref_y')
			if config.train_eps:
				trainable = trainable + ('eps',)
			if config.train_vi:
				trainable = trainable + ('vd',)
			params_flat, unflatten, params_fixed = bm.initialize(dim=dim, nbridges=config.nbridges, eta=config.init_eta, eps = config.init_eps,
				lfsteps=config.lfsteps, vdparams=vdparams_init, trainable=trainable)
			grad_and_loss = jax.jit(jax.grad(bm.compute_bound, 1, has_aux = True), static_argnums = (2, 3, 4))

			loss_fn = jax.jit(bm.compute_bound, static_argnums = (2, 3, 4))

		elif 'MCD' in config.boundmode:
			trainable = ('eta', 'gamma', 'mgridref_y')
			if config.train_eps:
				trainable = trainable + ('eps',)
			if config.train_vi:
				trainable = trainable + ('vd',)
			
			print(trainable)
			params_flat, unflatten, params_fixed = mcdbm.initialize(dim=dim, nbridges=config.nbridges, vdparams=vdparams_init, eta=config.init_eta, eps = config.init_eps,
				trainable=trainable, mode=config.boundmode)
			grad_and_loss = jax.jit(jax.grad(mcdbm.compute_bound, 1, has_aux = True), static_argnums = (2, 3, 4))

			loss_fn = jax.jit(mcdbm.compute_bound, static_argnums = (2, 3, 4))

		else:
			raise NotImplementedError('Mode %s not implemented.' % config.boundmode)

		# Average over 30 seeds, 500 samples each after training is done.
		n_samples = config.n_samples
		n_input_dist_seeds = config.n_input_dist_seeds

		if sample_from_target_fn is not None:
			target_samples = sample_from_target_fn(jax.random.PRNGKey(1), n_samples * n_input_dist_seeds)
		else:
			target_samples = None

		losses, diverged, params_flat, tracker = opt.run(config, config.lr, config.iters, params_flat, unflatten, params_fixed, log_prob_model, grad_and_loss,
			trainable, train_rng_key_gen, log_prefix='train', target_samples=target_samples)


		eval_losses, samples = opt.sample(
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
		
		# Plot samples
		if config.model in ["nice", "funnel"]:
			other_target_samples = sample_from_target_fn(jax.random.PRNGKey(2), samples.shape[0])

			w2_dists, self_w2_dists = [], []
			for i in range(n_input_dist_seeds):
				
				samples_i = samples[i * n_samples : (i + 1) * n_samples, ...]
				target_samples_i = target_samples[i * n_samples : (i + 1) * n_samples, ...]
				other_target_samples_i = other_target_samples[i * n_samples : (i + 1) * n_samples, ...]
				w2_dists.append(W2_distance(samples_i, target_samples_i))
				self_w2_dists.append(W2_distance(target_samples_i, other_target_samples_i))

			if config.model == "nice":
				make_grid(samples, config.im_size, n=64, wandb_prefix="images/sample")
			
			wandb.log({"w2_dist": onp.mean(onp.array(w2_dists)),
			  			"w2_dist_std": onp.std(onp.array(w2_dists)),
						"self_w2_dist": onp.mean(onp.array(self_w2_dists)),	
						"self_w2_dist_std": onp.std(onp.array(self_w2_dists))})

		params_train, params_notrain = unflatten(params_flat)
		params = {**params_train, **params_notrain}

		if config.wandb.log_artifact:
			artifact_name = f"{config.model}_{config.boundmode}_{config.nbridges}"

			artifact = wandb.Artifact(
				artifact_name, 
				type="nice_params",
				metadata={
				**{"alpha": config.alpha,
					"n_bits": config.n_bits,
					"im_size": config.im_size}
				})
			
			# Save model
			with artifact.new_file("params.pkl", "wb") as f:
				pickle.dump(params, f)
			
			wandb.log_artifact(artifact)


if __name__ == "__main__":
    import os
    import sys

    # if sys.argv:
    #     # pass wandb API as argv[1] and set environment variable
    #     # 'python mll_optim.py MY_API_KEY'
    os.environ["WANDB_API_KEY"] = "9835d6db89010f73306f92bb9a080c9751b25d28"

    # Adds jax flags to the program.
    jax.config.config_with_absl()

    # Snippet to pass configs into the main function.
    def _main(argv):
        del argv
        config = FLAGS.config
        main(config)

    app.run(_main)