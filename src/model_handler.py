from jax import grad, vmap
import jax.numpy as np
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


models_gym = ['lorenz', 'brownian', 'banana']

def load_model(model = 'log_sonar', config = None):
	if model in models_gym:
		return load_model_gym(model)
	if 'nice' in model:
		return load_model_nice(model, config)
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
		def _sample():
			return flow.sample(config.batch_size)
		return _logpx, (_logpx, _recons, _sample)
	
	forward = hk.multi_transform(forward_fn)

	logpx_fn, _, _ = forward.apply

	logpx_fn_without_rng = lambda x: np.squeeze(logpx_fn(loaded_params, jax.random.PRNGKey(1), x[None, :]))

	return logpx_fn_without_rng, config.im_size ** 2




		




