import jax.numpy as np
import jax
import variationaldist as vd
import momdist as md
from jax.flatten_util import ravel_pytree
import functools
import mcd_utils
from nn import initialize_mcd_network



def initialize(dim, vdparams=None, nbridges=0, eps=0.01, gamma = 10., eta = 0.5, ngridb=32, mgridref_y=None, trainable = ['eps'], use_score_nn=True,
	emb_dim=20, seed=1, mode="MCD_U_lp-e"):
	"""
	Modes allowed:
		- MCD_ULA: This is ULA. Method from Thin et al.
		- MCD_ULA_sn: This is MCD. Method from Doucet et al.
		- MCD_U_a-lp: UHA but with approximate sampling of momentum (no score network).
		- MCD_U_a-lp-sn: Approximate sampling of momentum, followed by leapfrog, using score network(x, rho) for backward sampling.
		- MCD_CAIS_sn: CAIS with trainable SN.
        - MCD_CAIS_UHA_sn: CAIS underdampened with trainable SN.
		"""
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

	if 'gamma' in trainable:
		params_train['gamma'] = gamma
	else:
		params_notrain['gamma'] = gamma

	if 'eta' in trainable:
		params_train['eta'] = eta
	else:
		params_notrain['eta'] = eta

	if mode in ["MCD_ULA_sn", "MCD_U_e-lp-sna", "MCD_U_a-lp-sna", "MCD_CAIS_sn"]:
		init_fun_sn, apply_fun_sn = initialize_mcd_network(dim, emb_dim, nbridges, nlayers=3)
		params_train['sn'] = init_fun_sn(jax.random.PRNGKey(seed), None)[1]
	elif mode in ["MCD_U_a-lp-sn", "MCD_U_ea-lp-sn", "MCD_U_a-nv-sn", "MCD_CAIS_UHA_sn"]:
		init_fun_sn, apply_fun_sn = initialize_mcd_network(dim, emb_dim, nbridges, rho_dim=dim, nlayers=3)
		params_train['sn'] = init_fun_sn(jax.random.PRNGKey(seed), None)[1]
	else:
		apply_fun_sn = None
		print("No score network needed by the method.")

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
	params_fixed = (dim, nbridges, mode, apply_fun_sn)
	params_flat, unflatten = ravel_pytree((params_train, params_notrain))
	return params_flat, unflatten, params_fixed


def compute_ratio(seed, params_flat, unflatten, params_fixed, log_prob):
	params_train, params_notrain = unflatten(params_flat)
	params_notrain = jax.lax.stop_gradient(params_notrain)
	params = {**params_train, **params_notrain} # Gets all parameters in single place
	dim, nbridges, _, _ = params_fixed

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
		z, w_mom, _ = mcd_utils.evolve(z, betas, params, rng_key, params_fixed, log_prob)
		w += w_mom

	# Update weight with final model evaluation
	w = w + log_prob(z)
	return -1. * w, (z, _)


# @functools.partial(jax.jit, static_argnums = (2, 3, 4))
def compute_bound(seeds, params_flat, unflatten, params_fixed, log_prob):
	ratios, (z, _) = jax.vmap(compute_ratio, in_axes = (0, None, None, None, None))(seeds, params_flat, unflatten, params_fixed, log_prob)
	# ratios, (z, _) = compute_ratio(seeds[0], params_flat, unflatten, params_fixed, log_prob)
	return ratios.mean(), (ratios, z)
