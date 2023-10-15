import jax
import jax.numpy as np
import variationaldist as vd


def evolve_underdamped_me_e(z, betas, params, rng_key_gen, params_fixed, log_prob_model, sample_kernel, log_prob_kernel):
    def U(z, beta):
        return -1. * (beta * log_prob_model(z) + (1. - beta) * vd.log_prob(params["vd"], z))

    def evolve(aux, i):
        z, rho, w, rng_key_gen = aux
        beta = betas[i]

        # Forward kernel
        fk_rho_mean = params["eta"] * rho
        scale = np.sqrt(1. - params["eta"] ** 2)

        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        rho_prime = sample_kernel(rng_key, fk_rho_mean, scale)

        rho_new = rho_prime - params["eps"] * jax.grad(U)(z, beta)
        z_new = z + params["eps"] * rho_new

        # Backwards kernel
        bk_rho_mean = params["eta"] * rho_prime + 2 * apply_fun_sn(params["sn"], z, i) # * (1. - params["eta"])
        # bk_rho_mean = params["eta"] * rho_prime

        # Evaluate kernels
        fk_log_prob = log_prob_kernel(rho_prime, fk_rho_mean, scale)
        bk_log_prob = log_prob_kernel(rho, bk_rho_mean, scale)

        # Update weight and return
        w += bk_log_prob - fk_log_prob

        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        aux = z_new, rho_new, w, rng_key_gen
        return aux, None

    dim, nbridges, damped, use_sn, apply_fun_sn = params_fixed
    # Sample initial momentum
    rng_key, rng_key_gen = jax.random.split(rng_key_gen)
    rho = jax.random.normal(rng_key, shape = (z.shape[0],)) # (dim,)

    # Add initial momentum term to w
    w = 0.
    w = w - log_prob_kernel(rho, np.zeros(z.shape[0]), 1.)

    # Evolve system
    rng_key, rng_key_gen = jax.random.split(rng_key_gen)
    aux = (z, rho, w, rng_key_gen)
    aux, _ = jax.lax.scan(evolve, aux, np.arange(nbridges))
    z, rho, w, _ = aux

    # Add final momentum term to w
    w = w + log_prob_kernel(rho, np.zeros(z.shape[0]), 1.)
    
    return z, w, None