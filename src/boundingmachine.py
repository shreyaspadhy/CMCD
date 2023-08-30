import jax.numpy as np
import jax
import variationaldist as vd
import momdist as md
from jax.flatten_util import ravel_pytree
import functools
import ais_utils



def initialize(dim, vdparams=None, nbridges=0, lfsteps=1, eps=0.0, eta=0.5, mdparams=None,
	ngridb=32, mgridref_y=None, trainable = ['eps', 'eta']):
	params_train = {} # Has all trainable parameters
	params_notrain = {} # Non trainable parameters

	if 'vd' in trainable:
		params_train['vd'] = vdparams
		if vdparams is None:
			params_train['vd'] = vd.initialize(dim)
	else:
		params_notrain['vd'] = vdparams
		if vdparams is None:
			params_notrain['vd'] = vd.initialize(dim)

	if 'eps' in trainable:
		params_train['eps'] = eps
	else:
		params_notrain['eps'] = eps

	if 'eta' in trainable:
		params_train['eta'] = eta
	else:
		params_notrain['eta'] = eta

	if 'md' in trainable:
		params_train['md'] = mdparams
		if mdparams is None:
			params_train['md'] = md.initialize(dim)
	else:
		params_notrain['md'] = mdparams
		if mdparams is None:
			params_notrain['md'] = md.initialize(dim)

	# Everything related to betas
	if mgridref_y is not None:
		ngridb = mgridref_y.shape[0] - 1
	else:
		if nbridges < ngridb:
			ngridb = nbridges
		mgridref_y = np.ones(ngridb + 1) * 1.
	params_notrain['gridref_x'] = np.linspace(0, 1, ngridb + 2)
	params_notrain['target_x'] = np.linspace(0, 1, nbridges + 2)[1:-1]
	if 'mgridref_y' in trainable:
		params_train['mgridref_y'] = mgridref_y
	else:
		params_notrain['mgridref_y'] = mgridref_y

	# Other fixed parameters
	params_fixed = (dim, nbridges, lfsteps)
	params_flat, unflatten = ravel_pytree((params_train, params_notrain))
	return params_flat, unflatten, params_fixed


def compute_ratio(seed, params_flat, unflatten, params_fixed, log_prob):
	params_train, params_notrain = unflatten(params_flat)
	params_notrain = jax.lax.stop_gradient(params_notrain)
	params = {**params_train, **params_notrain} # Gets all parameters in single place
	dim, nbridges, lfsteps = params_fixed

	if nbridges >= 1:
		gridref_y = np.cumsum(params['mgridref_y']) / np.sum(params['mgridref_y'])
		gridref_y = np.concatenate([np.array([0.]), gridref_y])
		betas = np.interp(params['target_x'], params['gridref_x'], gridref_y)

	rng_key_gen = jax.random.PRNGKey(seed)

	rng_key, rng_key_gen = jax.random.split(rng_key_gen)
	z = vd.sample_rep(rng_key, params['vd'])
	w = -vd.log_prob(params['vd'], z)

	# Evolve UHA and update weight
	delta_H = np.array([0.])
	if nbridges >= 1:
		rng_key, rng_key_gen = jax.random.split(rng_key_gen)
		z, w_mom, delta_H = ais_utils.evolve(z, betas, params, rng_key, params_fixed, log_prob)
		w += w_mom

	# Update weight with final model evaluation
	w = w + log_prob(z)
	delta_H = np.max(np.abs(delta_H))
	# delta_H = np.mean(np.abs(delta_H))
	return -1. * w, (z, delta_H)


# @functools.partial(jax.jit, static_argnums = (2, 3, 4))
def compute_bound(seeds, params_flat, unflatten, params_fixed, log_prob):
	ratios, (z, _) = jax.vmap(compute_ratio, in_axes = (0, None, None, None, None))(seeds, params_flat, unflatten, params_fixed, log_prob)
	return ratios.mean(), (ratios, z)




