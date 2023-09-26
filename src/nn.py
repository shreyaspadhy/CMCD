import jax
from jax.example_libraries.stax import Dense, serial, Softplus, FanInSum, FanOut, Identity, parallel
from jax.nn.initializers import glorot_normal, normal
from jax.nn import softplus
import jax.numpy as np



def initialize_embedding(rng, nbridges, emb_dim, factor=0.05):
	return jax.random.normal(rng, shape = (nbridges, emb_dim)) * factor


def initialize_mcd_network(x_dim, emb_dim, nbridges, rho_dim=0, nlayers=4):
	in_dim = x_dim + rho_dim + emb_dim


	layers = ([
		serial(FanOut(2), parallel(Identity, serial(Dense(in_dim), Softplus)), FanInSum),
		serial(FanOut(2), parallel(Identity, serial(Dense(in_dim), Softplus)), FanInSum),
		Dense(x_dim)
		])
	
	init_fun_nn, apply_fun_nn = serial(*layers)

	def init_fun(rng, input_shape):
		params = {}
		output_shape, params_nn = init_fun_nn(rng, (in_dim,))
		params["nn"] = params_nn
		rng, _ = jax.random.split(rng)
		params["emb"] = initialize_embedding(rng, nbridges, emb_dim)
		params["factor_sn"] = np.array(0.)
		return output_shape, params
	
	def apply_fun(params, inputs, i, **kwargs):
		# inputs has size (x_dim)
		emb = params["emb"][i, :] # (emb_dim,)
		input_all = np.concatenate([inputs, emb])
		return apply_fun_nn(params["nn"], input_all) * params["factor_sn"] # (x_dim,)

	return init_fun, apply_fun







