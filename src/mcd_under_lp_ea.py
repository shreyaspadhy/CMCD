import jax
import jax.numpy as np
import variationaldist as vd


def evolve_underdamped_lp_ea(z, betas, params, rng_key_gen, params_fixed, log_prob_model, sample_kernel, log_prob_kernel, use_sn=False, full_sn=True):
    def U(z, beta):
        return -1. * (beta * log_prob_model(z) + (1. - beta) * vd.log_prob(params["vd"], z))

    def evolve(aux, i):
        z, rho, w, rng_key_gen = aux
        beta = betas[i]

        # Forward kernel
        eta = np.exp(-params["gamma"] * params["eps"])
        eta_aux = params["gamma"] * params["eps"] # This better be close to zero...
        # fk_rho_mean = rho * (1. - eta_aux)
        scale = np.sqrt(2. * eta_aux)
        fk_rho_mean = rho * eta
        scale_f = np.sqrt(1. - eta ** 2)

        # print(eta)
        # exit()

        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        rho_prime = sample_kernel(rng_key, fk_rho_mean, scale_f)

        rho_prime_prime = rho_prime - params["eps"] * jax.grad(U)(z, beta) / 2.
        z_new = z + params["eps"] * rho_prime_prime
        rho_new = rho_prime_prime - params["eps"] * jax.grad(U)(z_new, beta) / 2.

        # Backwards kernel
        if not use_sn:
            bk_rho_mean = rho_prime * (1. - eta_aux)
        else:
            if not full_sn:
                bk_rho_mean = rho_prime * (1. - eta_aux) + 2 * eta_aux * apply_fun_sn(params["sn"], z, i) # No real reason for this, just to try
            else:
                input_sn = np.concatenate([z, rho_prime])
                bk_rho_mean = rho_prime * (1. - eta_aux) + 2 * eta_aux * apply_fun_sn(params["sn"], input_sn, i)

        # Evaluate kernels
        fk_log_prob = log_prob_kernel(rho_prime, fk_rho_mean, scale_f)
        bk_log_prob = log_prob_kernel(rho, bk_rho_mean, scale)

        # Update weight and return
        w += bk_log_prob - fk_log_prob

        # # print(fk_log_prob, bk_log_prob, scale, scale_f)
        # print("===================")
        # print(eta)
        # print("======================================")
        # print(eta_aux)
        # print("============================================================================")
        # print(w)
        # print("============================================================================")
        # print("============================================================================")
        # print("============================================================================")

        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        aux = z_new, rho_new, w, rng_key_gen
        return aux, None

    dim, nbridges, mode, apply_fun_sn = params_fixed
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

    # for i in range(nbridges):
    #     aux, _ = evolve(aux, i)

    z, rho, w, _ = aux

    # Add final momentum term to w
    w = w + log_prob_kernel(rho, np.zeros(z.shape[0]), 1.)
    
    return z, w, None