import vardist.diag_gauss as diagvd

def initialize(dim):
	return diagvd.initialize(dim)

def sample_rep(rng_key, vdparams):
	return diagvd.sample_rep(rng_key, vdparams)

def log_prob(vdparams, z):
	return diagvd.log_prob(z, vdparams)



