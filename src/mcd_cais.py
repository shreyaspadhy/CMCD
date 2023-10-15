import jax
import jax.numpy as np
import variationaldist as vd

import pdb

def evolve_overdamped_cais(z, betas, params, rng_key_gen, params_fixed, log_prob_model, sample_kernel, log_prob_kernel, 
                           use_sn=False, beta_schedule=None, grad_clipping=False):
    def U(z, beta):
        return -1. * (beta * log_prob_model(z) + (1. - beta) * vd.log_prob(params['vd'], z))

    def gradU(z, beta, clip=1e2):
        p =  lambda z:  vd.log_prob(params['vd'], z)
        gp = jax.grad(p)(z)
        u = lambda z: log_prob_model(z)
        gu = jax.grad(u)(z)
        guc = np.clip(gu, -clip, clip)
        return -1. * (beta * guc + (1. - beta) * gp)
    
    dim, nbridges, mode, apply_fun_sn = params_fixed


    def _eps_schedule(init_eps, i, final_eps=0.0001):
        # Implement linear decay b/w init_eps and final_eps
        return (final_eps - init_eps) / (nbridges - 1) * i + init_eps

    def _cosine_eps_schedule(init_eps, i, s=0.008):
        # Implement cosine decay b/w init_eps and final_eps
        phase = i / nbridges 

        decay = np.cos((phase + s) / (1 + s) * 0.5 * np.pi) ** 2

        return init_eps * decay
    

    def evolve(aux, i, stable=grad_clipping, beta_schedule=beta_schedule):
        z, w, rng_key_gen = aux
        beta = betas[i]

        # Forward kernel
#         fk_mean = z - params["eps"] * jax.grad(U)(z, beta) - params["eps"] * apply_fun_sn(params["sn"], z, i) # - because it is gradient of U = -log \pi
        uf = gradU(z, beta) if stable else jax.grad(U)(z, beta)

        if beta_schedule == 'cos_sq':
            eps = _cosine_eps_schedule(params["eps"], i)
        elif beta_schedule == 'linear':
            eps = _eps_schedule(params["eps"], i)
        else:
            eps = params["eps"]

        fk_mean = z - eps * uf - eps * apply_fun_sn(params["sn"], z, i)
        
        scale = np.sqrt(2 * eps)

        # Sample 
        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        z_new = sample_kernel(rng_key, fk_mean, scale)

        # Backwards kernel
        # ub = jax.grad(U)(z_new, beta)
        ub = gradU(z_new, beta) if stable else jax.grad(U)(z_new, beta)
        if not use_sn:
            bk_mean = z_new - eps * ub # Ignoring NN, assuming initialization, recovers method from Thin et al.
        else:
            bk_mean = z_new - eps * ub + eps * apply_fun_sn(params["sn"], z_new, i + 1) # Recovers Doucet et al.

        # Evaluate kernels
        fk_log_prob = log_prob_kernel(z_new, fk_mean, scale)
        bk_log_prob = log_prob_kernel(z, bk_mean, scale)

        # Update weight and return
        w += bk_log_prob - fk_log_prob
        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        aux = z_new, w, rng_key_gen
        return aux, None

    print(f'running CAIS')
    
    # Evolve system
    rng_key, rng_key_gen = jax.random.split(rng_key_gen)
    aux = (z, 0, rng_key_gen)
    aux, _ = jax.lax.scan(evolve, aux, np.arange(nbridges))
    
    z, w, _ = aux
    return z, w, None

