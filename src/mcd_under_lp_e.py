import jax
import jax.numpy as np
import variationaldist as vd


def evolve_underdamped_lp_e(
    z,
    betas,
    params,
    rng_key_gen,
    params_fixed,
    log_prob_model,
    sample_kernel,
    log_prob_kernel,
    use_sn=False,
):
    def U(z, beta):
        return -1.0 * (
            beta * log_prob_model(z) + (1.0 - beta) * vd.log_prob(params["vd"], z)
        )

    def evolve(aux, i):
        z, rho, w, rng_key_gen = aux
        beta = betas[i]

        # Forward kernel
        fk_rho_mean = params["eta"] * rho
        scale = np.sqrt(1.0 - params["eta"] ** 2)

        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        rho_prime = sample_kernel(rng_key, fk_rho_mean, scale)

        rho_prime_prime = rho_prime - params["eps"] * jax.grad(U)(z, beta) / 2.0
        z_new = z + params["eps"] * rho_prime_prime
        rho_new = rho_prime_prime - params["eps"] * jax.grad(U)(z_new, beta) / 2.0

        # Backwards kernel
        if not use_sn:
            bk_rho_mean = params["eta"] * rho_prime
        else:
            bk_rho_mean = params["eta"] * rho_prime + 2 * apply_fun_sn(
                params["sn"], z, i
            ) * (1.0 - params["eta"])

        # Evaluate kernels
        fk_log_prob = log_prob_kernel(rho_prime, fk_rho_mean, scale)
        bk_log_prob = log_prob_kernel(rho, bk_rho_mean, scale)

        # Update weight and return
        w += bk_log_prob - fk_log_prob

        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        aux = z_new, rho_new, w, rng_key_gen
        return aux, None

    dim, nbridges, mode, apply_fun_sn = params_fixed
    # Sample initial momentum
    rng_key, rng_key_gen = jax.random.split(rng_key_gen)
    rho = jax.random.normal(rng_key, shape=(z.shape[0],))  # (dim,)

    # Add initial momentum term to w
    w = 0.0
    w = w - log_prob_kernel(rho, np.zeros(z.shape[0]), 1.0)

    # Evolve system
    rng_key, rng_key_gen = jax.random.split(rng_key_gen)
    aux = (z, rho, w, rng_key_gen)
    aux, _ = jax.lax.scan(evolve, aux, np.arange(nbridges))
    z, rho, w, _ = aux

    # Add final momentum term to w
    w = w + log_prob_kernel(rho, np.zeros(z.shape[0]), 1.0)

    return z, w, None
