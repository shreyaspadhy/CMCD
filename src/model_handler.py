from jax import grad, vmap
import jax.numpy as np
import jax
import numpyro
from jax.flatten_util import ravel_pytree
import numpyro.distributions as npdists
import models.logistic_regression as model_lr
import models.seeds as model_seeds
import inference_gym.using_jax as gym


models_gym = ['lorenz', 'brownian', 'banana']

def load_model(model = 'log_sonar'):
	if model in models_gym:
		return load_model_gym(model)
	if 'nice' in model:
		return load_model_nice(model)
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


def load_model_nice(model = 'size=14_nbits=3_alpha=0.5'):
	pass





