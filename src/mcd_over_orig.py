import jax
import jax.numpy as np
import variationaldist as vd


def evolve_overdamped_orig(z, betas, params, rng_key_gen, params_fixed, log_prob_model, sample_kernel, log_prob_kernel, use_sn=False):
    def U(z, beta):
        return -1. * (beta * log_prob_model(z) + (1. - beta) * vd.log_prob(params['vd'], z))

    def evolve(aux, i):
        z, w, rng_key_gen = aux
        beta = betas[i]

        # Forward kernel
        fk_mean = z - params["eps"] * jax.grad(U)(z, beta) # - because it is gradient of U = -log \pi
        scale = np.sqrt(2 * params["eps"])

        # Sample 
        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        z_new = sample_kernel(rng_key, fk_mean, scale)

        # Backwards kernel
        if not use_sn:
            bk_mean = z_new - params["eps"] * jax.grad(U)(z_new, beta) # Ignoring NN, assuming initialization, recovers method from Thin et al.
        else:
            bk_mean = z_new - params["eps"] * jax.grad(U)(z_new, beta) + params["eps"] * apply_fun_sn(params["sn"], z_new, i) # Recovers Doucet et al.

        # Evaluate kernels
        fk_log_prob = log_prob_kernel(z_new, fk_mean, scale)
        bk_log_prob = log_prob_kernel(z, bk_mean, scale)

        # Update weight and return
        w += bk_log_prob - fk_log_prob
        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        aux = z_new, w, rng_key_gen
        return aux, None

    dim, nbridges, mode, apply_fun_sn = params_fixed
    # Evolve system
    rng_key, rng_key_gen = jax.random.split(rng_key_gen)
    aux = (z, 0, rng_key_gen)
    aux, _ = jax.lax.scan(evolve, aux, np.arange(nbridges))
    
    z, w, _ = aux
    return z, w, None

