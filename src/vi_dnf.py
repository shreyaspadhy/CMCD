import jax
import jax.numpy as np


def evolve_overdamped_dnf(
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
    # def U(z, beta):
    #     return -1.0 * (
    #         beta * log_prob_model(z) + (1.0 - beta) * vd.log_prob(params["vd"], z)
    #     )

    # def gradU(z, beta, clip=1e3):
    #     p = lambda z: vd.log_prob(params["vd"], z)
    #     gp = jax.grad(p)(z)
    #     u = lambda z: log_prob_model(z)
    #     gu = jax.grad(u)(z)
    #     guc = np.clip(gu, -clip, clip)
    #     return -1.0 * (beta * guc + (1.0 - beta) * gp)

    def evolve(aux, i, stable=False):
        z, w, rng_key_gen = aux
        beta = betas[i]

        # Forward kernel
        # 		fk_mean = z - params["eps"] * jax.grad(U)(z, beta) - params["eps"] * apply_fun_sn(params["sn"], z, i) # - because it is gradient of U = -log \pi
        uf = 0  # gradU(z, beta) if stable else jax.grad(U)(z, beta)
        fk_mean = (
            z - params["eps"] * uf - params["eps"] * apply_fun_sn(params["sn"], z, i)
        )

        scale = np.sqrt(2 * params["eps"])

        # Sample
        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        z_new = sample_kernel(rng_key, fk_mean, scale)

        # Backwards kernel
        # ub = jax.grad(U)(z_new, beta)
        ub = 0  # gradU(z_new, beta) if stable else jax.grad(U)(z_new, beta)
        if not use_sn:
            bk_mean = (
                z_new - params["eps"] * ub
            )  # Ignoring NN, assuming initialization, recovers method from Thin et al.
        else:
            bk_mean = (
                z_new
                - params["eps"] * ub
                + params["eps"] * apply_fun_sn(params["sn_2"], z_new, i + 1)
            )  # Recovers Doucet et al.

        # Evaluate kernels
        fk_log_prob = log_prob_kernel(z_new, fk_mean, scale)
        bk_log_prob = log_prob_kernel(z, bk_mean, scale)

        # Update weight and return
        w += bk_log_prob - fk_log_prob
        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        aux = z_new, w, rng_key_gen
        return aux, None

    print("running DNF")
    dim, nbridges, mode, apply_fun_sn = params_fixed
    # Evolve system
    rng_key, rng_key_gen = jax.random.split(rng_key_gen)
    aux = (z, 0, rng_key_gen)
    aux, _ = jax.lax.scan(evolve, aux, np.arange(nbridges))

    z, w, _ = aux
    return z, w, None
