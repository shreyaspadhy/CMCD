import jax
import jax.numpy as np
import numpyro.distributions as npdist
import variationaldist as vd
import momdist as md
from mcd_over_orig import evolve_overdamped_orig
from mcd_under_me_e import evolve_underdamped_me_e
from mcd_under_lp_e import evolve_underdamped_lp_e
from mcd_under_lp_a import evolve_underdamped_lp_a
from mcd_under_lp_a_cais import evolve_underdamped_lp_a_cais
from mcd_under_lp_ea import evolve_underdamped_lp_ea
from mcd_cais import evolve_overdamped_cais


# For transition kernel
def sample_kernel(rng_key, mean, scale):
	eps = jax.random.normal(rng_key, shape = (mean.shape[0],))
	return mean + scale * eps

def log_prob_kernel(x, mean, scale):
	dist = npdist.Independent(npdist.Normal(loc=mean, scale=scale), 1)
	return dist.log_prob(x)

def evolve(z, betas, params, rng_key_gen, params_fixed, log_prob_model):
	mode = params_fixed[2]
	if mode == "MCD_ULA":
		return evolve_overdamped_orig(z, betas, params, rng_key_gen, params_fixed, log_prob_model, sample_kernel, log_prob_kernel, use_sn=False)
	elif mode == "MCD_ULA_sn":
		return evolve_overdamped_orig(z, betas, params, rng_key_gen, params_fixed, log_prob_model, sample_kernel, log_prob_kernel, use_sn=True)
	elif mode == "MCD_U_e-lp":
		return evolve_underdamped_lp_e(z, betas, params, rng_key_gen, params_fixed, log_prob_model, sample_kernel, log_prob_kernel, use_sn=False)
	elif mode == "MCD_U_e-lp-sna":
		return evolve_underdamped_lp_e(z, betas, params, rng_key_gen, params_fixed, log_prob_model, sample_kernel, log_prob_kernel, use_sn=True)
	elif mode == "MCD_U_a-lp":
		return evolve_underdamped_lp_a(z, betas, params, rng_key_gen, params_fixed, log_prob_model, sample_kernel, log_prob_kernel, use_sn=False)
	elif mode == "MCD_U_a-lp-sna":
		return evolve_underdamped_lp_a(z, betas, params, rng_key_gen, params_fixed, log_prob_model, sample_kernel, log_prob_kernel, use_sn=True,
			full_sn=False)
	elif mode == "MCD_U_a-lp-sn":
		return evolve_underdamped_lp_a(z, betas, params, rng_key_gen, params_fixed, log_prob_model, sample_kernel, log_prob_kernel, use_sn=True,
			full_sn=True)
	elif mode == "MCD_U_ea-lp-sn":
		return evolve_underdamped_lp_ea(z, betas, params, rng_key_gen, params_fixed, log_prob_model, sample_kernel, log_prob_kernel, use_sn=True,
			full_sn=True)
	elif mode == "MCD_CAIS_sn":
		return evolve_overdamped_cais(z, betas, params, rng_key_gen, params_fixed, log_prob_model, sample_kernel, log_prob_kernel, use_sn=True)
	elif mode == "MCD_CAIS_UHA_sn":
		return evolve_underdamped_lp_a_cais(z, betas, params, rng_key_gen, params_fixed, log_prob_model, sample_kernel, log_prob_kernel, use_sn=True,
			full_sn=True)
	else:
		raise NotImplementedError("Mode not implemented.")


