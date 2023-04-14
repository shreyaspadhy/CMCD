# import jax.numpy as np
import numpy as onp
# import sklearn.preprocessing



def pad_with_const(X):
	extra = onp.ones((X.shape[0], 1))
	return onp.hstack([extra, X])

# def standardize_and_pad(X):
# 	tform = sklearn.preprocessing.StandardScaler()
# 	return pad_with_const(tform.fit_transform(X))

def standardize_and_pad(X):
	mean = onp.mean(X, axis = 0)
	std = onp.std(X, axis = 0)
	std[std == 0] = 1.
	X = (X - mean) / std
	return pad_with_const(X)