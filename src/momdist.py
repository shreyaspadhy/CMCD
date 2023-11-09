import jax
import jax.numpy as np
import numpyro.distributions as npdist

# For now, only diagonal


def initialize(dim):
    # For now only diagonal, parameterize by logscale parameters - only returns scale parameters, mean always zero
    return np.zeros(dim)


def sample(rng_key, eta, prev, params):
    # Params is just an array with logscale parameters
    dim = params.shape[0]
    rho_indep = np.exp(params) * jax.random.normal(rng_key, (dim,))
    if prev is None:
        rho = rho_indep
    else:
        rho = eta * prev + np.sqrt(1.0 - eta**2) * rho_indep
    return rho


def log_prob(rho, params):
    # Params is just an array with logscale parameters
    dim = rho.shape[0]
    dist = npdist.Independent(npdist.Normal(loc=np.zeros(dim), scale=np.exp(params)), 1)
    return dist.log_prob(rho)
