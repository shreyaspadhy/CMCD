import numpyro
import pickle
import jax.numpy as np
import numpyro.distributions as pydist
from .data_utils import standardize_and_pad


def load_data(dset):
	if dset == 'australian':
		with open('models/datasets/australian_full.pkl', 'rb') as f:
			X, Y = pickle.load(f)
		Y = (Y + 1) // 2
	if dset == 'ionosphere':
		with open('models/datasets/ionosphere_full.pkl', 'rb') as f:
			X, Y = pickle.load(f)
		Y = (Y + 1) // 2
	if dset == 'sonar':
		with open('models/datasets/sonar_full.pkl', 'rb') as f:
			X, Y = pickle.load(f)
		Y = (Y + 1) // 2
	if dset == 'a1a':
		with open('models/datasets/a1a_full.pkl', 'rb') as f:
			X, Y = pickle.load(f)
		Y = (Y + 1) // 2
	if dset == 'madelon':
		with open('models/datasets/madelon_full.pkl', 'rb') as f:
			X, Y = pickle.load(f)
		Y = (Y + 1) // 2
	X = standardize_and_pad(X)
	return X, Y


def load_model(dset):
	def model(Y):
		w = numpyro.sample("weights", pydist.Normal(np.zeros(dim), np.ones(dim)))
		logits = np.dot(X, w)
		with numpyro.plate('J', n_data):
			y = numpyro.sample("y", pydist.BernoulliLogits(logits), obs=Y)
	X, Y = load_data(dset)
	dim = X.shape[1]
	n_data = X.shape[0]
	model_args = (Y,)
	return model, model_args



