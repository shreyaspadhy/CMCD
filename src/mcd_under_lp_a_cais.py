import jax
import jax.numpy as np
import variationaldist as vd


def evolve_underdamped_lp_a_cais(z, betas, params, rng_key_gen, params_fixed, log_prob_model, sample_kernel, log_prob_kernel, use_sn=False, full_sn=True):
    def U(z, beta):
        return -1. * (beta * log_prob_model(z) + (1. - beta) * vd.log_prob(params["vd"], z))

    def gradU(z, beta, clip=1e2):
        p =  lambda z:  vd.log_prob(params['vd'], z)
        gp = jax.grad(p)(z)
        u = lambda z: log_prob_model(z)
        gu = jax.grad(u)(z)
        guc = np.clip(gu, -clip, clip)
        return -1. * (beta * guc + (1. - beta) * gp)

    dim, nbridges, mode, apply_fun_sn = params_fixed

    def _cosine_eps_schedule(init_eps, i, s=0.008):
        # Implement cosine decay b/w init_eps and final_eps
        phase = i / nbridges 

        decay = np.cos((phase + s) / (1 + s) * 0.5 * np.pi) ** 2

        return init_eps * decay

    def evolve(aux, i, stable=True):
        z, rho, w, rng_key_gen = aux
        beta = betas[i]

        # Forward kernel
        uf = gradU(z, beta) if stable else jax.grad(U)(z, beta)

        eps = _cosine_eps_schedule(params["eps"], i)

        eta_aux = params["gamma"] * eps 
        input_sn_old = np.concatenate([z, rho])
        fk_rho_mean = rho * (1. - eta_aux) - 2. * eta_aux * apply_fun_sn(params["sn"], input_sn_old, i)

        scale = np.sqrt(2. * eta_aux) 

        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        rho_prime = sample_kernel(rng_key, fk_rho_mean, scale)


        # Leap frog step
        rho_prime_prime = rho_prime - eps * uf / 2.
        z_new = z + eps * rho_prime_prime

        ub = gradU(z_new, beta) if stable else jax.grad(U)(z_new, beta)

        rho_new = rho_prime_prime - eps * ub / 2.

        # Backwards kernel
#         if not use_sn:
#         bk_rho_mean = rho_prime * (1. - eta_aux)
#         else:
#             if not full_sn:
#                 bk_rho_mean = rho_prime * (1. - eta_aux) + 2 * eta_aux * apply_fun_sn(params["sn"], z, i) # No real reason for this, just to try
#             else:
#                 
        input_sn = np.concatenate([z, rho_prime])
        bk_rho_mean = rho_prime * (1. - eta_aux) + 2. * eta_aux * apply_fun_sn(params["sn"], input_sn, i)

        # Evaluate kernels
        fk_log_prob = log_prob_kernel(rho_prime, fk_rho_mean, scale)
        bk_log_prob = log_prob_kernel(rho, bk_rho_mean, scale)

        # Update weight and return
        w += bk_log_prob - fk_log_prob

        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        aux = z_new, rho_new, w, rng_key_gen
        return aux, None

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