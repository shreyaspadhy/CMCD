import variationaldist as vd
import momdist as md
import jax
import jax.numpy as np
import numpyro.distributions as npdist



def evolve(z, betas, params, rng_key_gen, params_fixed, log_prob):
    def U(z, beta):
        return -1. * (beta * log_prob(z) + (1. - beta) * vd.log_prob(params['vd'], z))

    def evolve_bridges(aux, i):
        z, rho_prev, w, rng_key_gen = aux
        beta = betas[i]
        # Re-sample momentum
        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        rho = md.sample(rng_key, params['eta'], rho_prev, params['md'])
        # Simulate dynamics
        z_new, rho_new, delta_H = leapfrog(z, rho, beta)
        # Update weight
        w = w + md.log_prob(rho_new, params['md']) - md.log_prob(rho, params['md'])
        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        aux = z_new, rho_new, w, rng_key_gen
        # return aux, z2_new
        return aux, delta_H

    def leapfrog(z, rho, beta):
        def K(rho):
            return -1. * md.log_prob(rho, params['md'])

        def full_leap(aux, i):
            z, rho = aux
            rho = rho - params['eps'] * jax.grad(U, 0)(z, beta)
            z = z + params['eps'] * jax.grad(K, 0)(rho)
            aux = (z, rho)
            return aux, None

        # Half step for momentum
        U_init, U_grad = jax.value_and_grad(U, 0)(z, beta)
        rho = rho - params['eps'] * U_grad / 2.
        # Full step for z
        K_init, K_grad = jax.value_and_grad(K, 0)(rho)
        z = z + params['eps'] * K_grad
        
        # # Alternate full steps
        if lfsteps > 1:
            aux = (z, rho)
            aux = jax.lax.scan(full_leap, aux, np.arange(lfsteps - 1))[0]
            z, rho = aux

        # Half step for momentum
        U_final, U_grad = jax.value_and_grad(U, 0)(z, beta)
        rho = rho - params['eps'] * U_grad / 2.
        K_final = K(rho)

        delta_H = U_init + K_init - U_final - K_final

        return z, rho, delta_H

    nbridges = params_fixed[1]
    lfsteps = params_fixed[2]
    # Sample initial momentum
    rng_key, rng_key_gen = jax.random.split(rng_key_gen)
    rho = md.sample(rng_key, params['eta'], None, params['md'])
    # Evolve system
    rng_key, rng_key_gen = jax.random.split(rng_key_gen)
    aux = (z, rho, 0, rng_key_gen)
    aux, delta_H = jax.lax.scan(evolve_bridges, aux, np.arange(nbridges))
    z, _, w, _ = aux
    return z, w, delta_H




