import vardist.diag_gauss as diagvd

def initialize(dim, init_sigma=1.):
    return diagvd.initialize(dim, init_sigma=init_sigma)

def sample_rep(rng_key, vdparams):
    return diagvd.sample_rep(rng_key, vdparams)

def log_prob(vdparams, z):
    return diagvd.log_prob(z, vdparams)



