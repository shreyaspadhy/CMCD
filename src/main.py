import numpyro
import numpyro.distributions as dist
import jax.numpy as np
import jax
from jax.flatten_util import ravel_pytree
import argparse
import boundingmachine as bm
import iwboundingmachine as iwbm
import mcdboundingmachine as mcdbm
import opt
from model_handler import load_model
import pickle


savedir = '/mnt/nfs/work1/domke/tgeffner/AIS_score/results/'

args_parser = argparse.ArgumentParser(description='Process arguments')
args_parser.add_argument('-boundmode', type=str, default='UHA', help='Determines what method to use, see main.py.')
args_parser.add_argument('-model', type=str, default='log_sonar', help='Model to use.')
args_parser.add_argument('-N', type=int, default=5, help='Number of samples to estimate gradient at each step.')
args_parser.add_argument('-nbridges', type=int, default=10, help='Number of bridging densities.')
args_parser.add_argument('-lfsteps', type=int, default=1, help='Leapfrog steps, for UHA.')
args_parser.add_argument('-iters', type=int, default=15000, help='Number of iterations.')
args_parser.add_argument('-lr', type=float, default=0.01, help='Learning rate.')
args_parser.add_argument('-seed', type=int, default=1, help='Random seed to use.')
args_parser.add_argument('-id', type=int, default=-1, help='Unique ID for each run.')
args_parser.add_argument('-run_cluster', type=int, default=0, help='1: Running on cluster, 0: no cluster (for plotting, etc).')
info = args_parser.parse_args()

# Boundmodes
# 	- ULA uses MCD_ULA
# 	- MCD uses MCD_ULA_sn
#	- UHA uses UHA
# 	- LDVI uses MCD_U_a-lp-sn

iters_base=info.iters
log_prob_model, dim = load_model(info.model)
rng_key_gen = jax.random.PRNGKey(info.seed)

# Train initial variational distribution to maximize the ELBO
trainable=('vd',)
params_flat, unflatten, params_fixed = bm.initialize(dim=dim, nbridges=0, trainable=trainable)
grad_and_loss = jax.jit(jax.grad(bm.compute_bound, 1, has_aux = True), static_argnums = (2, 3, 4))
losses, diverged, params_flat, tracker = opt.run(info, 0.01, iters_base, params_flat, unflatten, params_fixed,
	log_prob_model, grad_and_loss, trainable, rng_key_gen)
vdparams_init = unflatten(params_flat)[0]['vd']

elbo_init = -np.mean(np.array(losses[-500:]))
print('Done training initial parameters, got ELBO %.2f.' % elbo_init)

if info.boundmode == 'UHA':
	trainable = ('vd', 'eps', 'eta', 'mgridref_y')
	params_flat, unflatten, params_fixed = bm.initialize(dim=dim, nbridges=info.nbridges, eta=0.0, eps = 0.00001,
		lfsteps=info.lfsteps, vdparams=vdparams_init, trainable=trainable)
	grad_and_loss = jax.jit(jax.grad(bm.compute_bound, 1, has_aux = True), static_argnums = (2, 3, 4))

elif 'MCD' in info.boundmode:
	trainable = ('vd', 'eps', 'eta', 'gamma', 'mgridref_y')
	params_flat, unflatten, params_fixed = mcdbm.initialize(dim=dim, nbridges=info.nbridges, vdparams=vdparams_init, eta=0.0, eps = 0.00001,
		trainable=trainable, mode=info.boundmode)
	grad_and_loss = jax.jit(jax.grad(mcdbm.compute_bound, 1, has_aux = True), static_argnums = (2, 3, 4))

else:
	raise NotImplementedError('Mode %s not implemented.' % info.boundmode)

losses, diverged, params_flat, tracker = opt.run(info, info.lr, info.iters, params_flat, unflatten, params_fixed, log_prob_model, grad_and_loss,
	trainable, rng_key_gen)

final_elbo = -np.mean(np.array(losses[-500:]))
print('Done training, got ELBO %.2f.' % final_elbo)

tracker['elbo_init'] = elbo_init
tracker['elbo_final'] = final_elbo
tracker["diverged"] = diverged

print(elbo_init, final_elbo)

params_train, params_notrain = unflatten(params_flat)
params = {**params_train, **params_notrain}

