from jax import grad, vmap
import jax.numpy as np
import numpy as onp
import jax.random as jr
import jax.scipy.linalg as slinalg
import jax
import numpyro
from jax.flatten_util import ravel_pytree
import numpyro.distributions as npdists
import models.logistic_regression as model_lr
import models.seeds as model_seeds
import inference_gym.using_jax as gym
import wandb
import pickle
import haiku as hk
from nice import NICE
import chex

from jax.scipy.special import logsumexp
from jax.scipy.stats import multivariate_normal
from jax.scipy.stats import norm
from typing import Any

import cp_utils
from annealed_flow_transport.densities import LogDensity


# TypeDefs
Array = np.ndarray
ConfigDict = Any

models_gym = ['lorenz', 'brownian', 'banana']

def load_model(model = 'log_sonar', config = None):
    if model in models_gym:
      return load_model_gym(model)
    if 'nice' in model:
      return load_model_nice(model, config)
    if 'funnel' in model:
      return load_model_funnel(model, config)
    if 'lgcp' in model:
      return load_model_lgcp(model, config)
    if 'gmm' in model:
      return load_model_gmm(model, config)

    return load_model_other(model)


def load_model_gym(model='banana'):
	def log_prob_model(z):
		x = target.default_event_space_bijector(z)
		return (target.unnormalized_log_prob(x) + target.default_event_space_bijector.forward_log_det_jacobian(z, event_ndims = 1))
	if model == 'lorenz':
		target = gym.targets.ConvectionLorenzBridge()
	if model == 'brownian':
		target = gym.targets.BrownianMotionUnknownScalesMissingMiddleObservations()
	if model == 'banana':
		target = gym.targets.Banana()
	target = gym.targets.VectorModel(target, flatten_sample_transformations=True)
	dim = target.event_shape[0]
	return log_prob_model, dim


def load_model_other(model = 'log_sonar'):
	if model == 'log_sonar':
		model, model_args = model_lr.load_model('sonar')
	if model == 'log_ionosphere':
		model, model_args = model_lr.load_model('ionosphere')
	if model == 'seeds':
		model, model_args = model_seeds.load_model()
	
	rng_key = jax.random.PRNGKey(1)
	model_param_info, potential_fn, constrain_fn, _ = numpyro.infer.util.initialize_model(rng_key, model, model_args = model_args)
	params_flat, unflattener = ravel_pytree(model_param_info[0])
	log_prob_model = lambda z: -1. * potential_fn(unflattener(z))
	dim = params_flat.shape[0]
	unflatten_and_constrain = lambda z: constrain_fn(unflattener(z))
	return log_prob_model, dim


def load_model_nice(model = 'nice', config = None):
	artifact_name = f"{config.alpha}_{config.n_bits}_{config.im_size}"

	api = wandb.Api()

	artifact = api.artifact(f"shreyaspadhy/cais/{artifact_name}:latest")
	loaded_params = pickle.load(open(artifact.file(), "rb"))

	def forward_fn():
		flow = NICE(config.im_size ** 2, h_dim=config.hidden_dim)

		def _logpx(x):
			return flow.logpx(x)
		def _recons(x):
			return flow.reverse(flow.forward(x))
		def _sample(n):
			return flow.sample(n)
		return _logpx, (_logpx, _recons, _sample)
	
	forward = hk.multi_transform(forward_fn)

	logpx_fn, _, sample_fn = forward.apply

	logpx_fn_without_rng = lambda x: np.squeeze(logpx_fn(loaded_params, jax.random.PRNGKey(1), x[None, :]))

	sample_fn_clean = lambda rng, n: sample_fn(loaded_params, rng, n)

	return logpx_fn_without_rng, config.im_size ** 2, sample_fn_clean


def load_model_funnel(model = 'funnel', config = None):

	d=config.funnel_d
	sig=config.funnel_sig
	clip_y=config.funnel_clipy

	def neg_energy(x):
		def unbatched(x):
			v = x[0]
			log_density_v = norm.logpdf(v,
										loc=0.,
										scale=3.)
			variance_other = np.exp(v)
			other_dim = d - 1
			cov_other = np.eye(other_dim) * variance_other
			mean_other = np.zeros(other_dim)
			log_density_other = multivariate_normal.logpdf(x[1:],
															mean=mean_other,
															cov=cov_other)
			return log_density_v + log_density_other
		output = np.squeeze(jax.vmap(unbatched)(x[None, :]))
		return output

	def sample_data(rng, n_samples):
		# sample from Nd funnel distribution

		y_rng, x_rng = jr.split(rng)

		y = (sig * jr.normal(y_rng, (n_samples, 1))).clip(-clip_y, clip_y)
		x = jr.normal(x_rng, (n_samples, d - 1)) * np.exp(-y / 2)
		return np.concatenate((y, x), axis=1)
	
	return neg_energy, d, sample_data


class ChallengingTwoDimensionalMixture(LogDensity):
  """A challenging mixture of Gaussians in two dimensions.

  num_dim should be 2. config is unused in this case.
  """

  def _check_constructor_inputs(self, config: ConfigDict,
                                sample_shape):
    del config
    # chex.assert_trees_all_equal(sample_shape, (2,))

  def raw_log_density(self, x: Array) -> Array:
    """A raw log density that we will then symmetrize."""
    mean_a = np.array([3.0, 0.])
    mean_b = np.array([-2.5, 0.])
    mean_c = np.array([2.0, 3.0])
    means = np.stack((mean_a, mean_b, mean_c), axis=0)
    cov_a = np.array([[0.7, 0.], [0., 0.05]])
    cov_b = np.array([[0.7, 0.], [0., 0.05]])
    cov_c = np.array([[1.0, 0.95], [0.95, 1.0]])
    covs = np.stack((cov_a, cov_b, cov_c), axis=0)
    log_weights = np.log(np.array([1./3, 1./3., 1./3.]))

    print(means.shape, covs.shape, x.shape)
    l = np.linalg.cholesky(covs)
    # y = np.linalg.solve(l, (x[None, :] - means))
    y = slinalg.solve_triangular(l, x[None, :] - means, lower=True, trans=0)
    mahalanobis_term = -1/2 * np.einsum("...i,...i->...", y, y)
    n = means.shape[-1]
    normalizing_term = -n / 2 * np.log(2 * np.pi) - np.log(
        l.diagonal(axis1=-2, axis2=-1)).sum(axis=1)
    individual_log_pdfs = mahalanobis_term + normalizing_term
    mixture_weighted_pdfs = individual_log_pdfs + log_weights
    return logsumexp(mixture_weighted_pdfs)

  def make_2d_invariant(self, log_density, x: Array) -> Array:
    density_a = log_density(x)
    density_b = log_density(np.flip(x))
    return np.logaddexp(density_a, density_b) - np.log(2)

  def evaluate_log_density(self, x: Array) -> Array:
    print(x.shape)
    density_func = lambda x: self.make_2d_invariant(self.raw_log_density, x)
    return density_func(x)
    # else:
    #   return jax.vmap(density_func)(x)
  
  def sample(self, rng_key, num_samples):
    mean_a = np.array([3.0, 0.0])
    mean_b = np.array([-2.5, 0.0])
    mean_c = np.array([2.0, 3.0])
    cov_a = np.array([[0.7, 0.0], [0.0, 0.05]])
    cov_b = np.array([[0.7, 0.0], [0.0, 0.05]])
    cov_c = np.array([[1.0, 0.95], [0.95, 1.0]])
    means = [mean_a, mean_b, mean_c]
    covs = [cov_a, cov_b, cov_c]
    log_weights = np.log(np.array([1.0 / 3, 1.0 / 3, 1.0 / 3]))
    num_components = len(means)
    samples = []
    k1, k2 = jr.split(rng_key)
    # Sample from the GMM components based on the mixture weights
    for i, _ in enumerate(range(num_samples)):
        # Sample a component index based on the mixture weights
        component_idx = jax.random.choice(k1 + i, num_components, p=np.exp(log_weights))
        # Sample from the chosen component
        chosen_mean = means[component_idx]
        chosen_cov = covs[component_idx]
        sample = jax.random.multivariate_normal(k2 + i, chosen_mean, chosen_cov)
        samples.append(sample)
    return np.stack(samples)



def load_model_gmm(model = "gmm", config = None):
  gmm = ChallengingTwoDimensionalMixture(config, sample_shape=(2,))
  
  # log_density_fn = lambda x: np.squeeze(gmm.evaluate_log_density(x[None, :]))

  # x = np.array([0., 0.])
  # print(x.shape)
  # print(gmm.evaluate_log_density(np.array([0., 0.])))
  return gmm.evaluate_log_density, 2, gmm.sample


class LogGaussianCoxPines(LogDensity):
  """Log Gaussian Cox process posterior in 2D for pine saplings data.

  This follows Heng et al 2020 https://arxiv.org/abs/1708.08396 .

  config.file_path should point to a csv file of num_points columns
  and 2 rows containg the Finnish pines data.

  config.use_whitened is a boolean specifying whether or not to use a
  reparameterization in terms of the Cholesky decomposition of the prior.
  See Section G.4 of https://arxiv.org/abs/2102.07501 for more detail.
  The experiments in the paper have this set to False.

  num_dim should be the square of the lattice sites per dimension.
  So for a 40 x 40 grid num_dim should be 1600.
  """

  def __init__(self,
               config: ConfigDict,
               num_dim: int = 1600):
    super().__init__(config, num_dim)

    # Discretization is as in Controlled Sequential Monte Carlo
    # by Heng et al 2017 https://arxiv.org/abs/1708.08396
    self._num_latents = num_dim
    self._num_grid_per_dim = int(np.sqrt(num_dim))

    bin_counts = np.array(
        cp_utils.get_bin_counts(self.get_pines_points(config.file_path),
                                self._num_grid_per_dim))

    self._flat_bin_counts = np.reshape(bin_counts, (self._num_latents))

    # This normalizes by the number of elements in the grid
    self._poisson_a = 1./self._num_latents
    # Parameters for LGCP are as estimated in Moller et al, 1998
    # "Log Gaussian Cox processes" and are also used in Heng et al.

    self._signal_variance = 1.91
    self._beta = 1./33

    self._bin_vals = cp_utils.get_bin_vals(self._num_grid_per_dim)

    def short_kernel_func(x, y):
      return cp_utils.kernel_func(x, y, self._signal_variance,
                                  self._num_grid_per_dim, self._beta)

    self._gram_matrix = cp_utils.gram(short_kernel_func, self._bin_vals)
    self._cholesky_gram = np.linalg.cholesky(self._gram_matrix)
    self._white_gaussian_log_normalizer = -0.5 * self._num_latents * np.log(
        2. * np.pi)

    half_log_det_gram = np.sum(np.log(np.abs(np.diag(self._cholesky_gram))))
    self._unwhitened_gaussian_log_normalizer = -0.5 * self._num_latents * np.log(
        2. * np.pi) - half_log_det_gram
    # The mean function is a constant with value mu_zero.
    self._mu_zero = np.log(126.) - 0.5*self._signal_variance

    if self._config.use_whitened:
      self._posterior_log_density = self.whitened_posterior_log_density
    else:
      self._posterior_log_density = self.unwhitened_posterior_log_density

  def  _check_constructor_inputs(self, config: ConfigDict, num_dim: int):
    expected_members_types = [("use_whitened", bool)]
    self._check_members_types(config, expected_members_types)
    num_grid_per_dim = int(np.sqrt(num_dim))
    if num_grid_per_dim * num_grid_per_dim != num_dim:
      msg = ("num_dim needs to be a square number for LogGaussianCoxPines "
             "density.")
      raise ValueError(msg)

    if not config.file_path:
      msg = "Please specify a path in config for the Finnish pines data csv."
      raise ValueError(msg)

  def get_pines_points(self, file_path):
    """Get the pines data points."""
    with open(file_path, mode="rt") as input_file:
    # with open(file_path, "rt") as input_file:
      b = onp.genfromtxt(input_file, delimiter=",")
    return b

  def whitened_posterior_log_density(self, white: Array) -> Array:
    quadratic_term = -0.5 * np.sum(white**2)
    prior_log_density = self._white_gaussian_log_normalizer + quadratic_term
    latent_function = cp_utils.get_latents_from_white(white, self._mu_zero,
                                                      self._cholesky_gram)
    log_likelihood = cp_utils.poisson_process_log_likelihood(
        latent_function, self._poisson_a, self._flat_bin_counts)
    return prior_log_density + log_likelihood

  def unwhitened_posterior_log_density(self, latents: Array) -> Array:
    white = cp_utils.get_white_from_latents(latents, self._mu_zero,
                                            self._cholesky_gram)
    prior_log_density = -0.5 * np.sum(
        white * white) + self._unwhitened_gaussian_log_normalizer
    log_likelihood = cp_utils.poisson_process_log_likelihood(
        latents, self._poisson_a, self._flat_bin_counts)
    return prior_log_density + log_likelihood

  def evaluate_log_density(self, x: Array) -> Array:
    # import pdb; pdb.set_trace()
    if len(x.shape) == 1:
      return self._posterior_log_density(x)
    else:
      return jax.vmap(self._posterior_log_density)(x)

def load_model_lgcp(model = 'lgcp', config = None):
	lgcp = LogGaussianCoxPines(config, num_dim=1600)
    

	return lgcp.evaluate_log_density, lgcp._num_latents