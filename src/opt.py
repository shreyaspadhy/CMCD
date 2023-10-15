import jax.numpy as np
import numpy as onp
import jax
from jax.flatten_util import ravel_pytree
from tqdm import tqdm
import sys
import functools
import wandb
from utils import make_grid, W2_distance, plot_gmm

def adam(step_size, b1 = 0.9, b2 = 0.999, eps = 1e-8):
    # Basically JAX's thing with added projection for some parameters.
    # Assumes ravel_pytree will always work the same way, so no need to update the
    # unflatten function (which may be problematic for jitting stuff)
    def init(x0):
        m0 = np.zeros_like(x0)
        v0 = np.zeros_like(x0)
        return x0, m0, v0
    def update(i, g, state, unflatten, trainable):
        def project(x, unflatten, trainable):
            x_train, x_notrain = unflatten(x)
            if 'eps' in trainable:
                x_train['eps'] = np.clip(x_train['eps'], 0.0000001, 0.5)
            if 'eta' in trainable:
                x_train['eta'] = np.clip(x_train['eta'], 0, 0.99)
            if 'gamma' in trainable:
                x_train['gamma'] = np.clip(x_train['gamma'], 0.001, None)
            if 'mgridref_y' in trainable:
                x_train['mgridref_y'] = jax.nn.relu(x_train['mgridref_y'] - 0.001) + 0.001
            return ravel_pytree((x_train, x_notrain))[0]
        x, m, v = state
        m = (1 - b1) * g + b1 * m # First moment estimate
        v = (1 - b2) * np.square(g) + b2 * v # Second moment estimate
        mhat = m / (1 - np.asarray(b1, m.dtype) ** (i + 1)) # Bias correction
        vhat = v / (1 - np.asarray(b2, m.dtype) ** (i + 1))
        x = x - step_size * mhat / (np.sqrt(vhat) + eps)
        x = project(x, unflatten, trainable)
        return x, m, v
    def get_params(state):
        x, _, _ = state
        return x
    return init, update, get_params

@functools.partial(jax.jit, static_argnums = (1, 2))
def collect_eps(params_flat, unflatten, trainable):
    if 'eps' in trainable:
        return unflatten(params_flat)[0]['eps']
    else:
        return unflatten(params_flat)[1]['eps']
    # return 0.

@functools.partial(jax.jit, static_argnums = (1, 2))
def collect_gamma(params_flat, unflatten, trainable):
    if 'gamma' in trainable:
        return unflatten(params_flat)[0]['gamma']
    return 0.

def run(info, lr, iters, params_flat, unflatten, params_fixed, log_prob_model, grad_and_loss, trainable, rng_key_gen, 
        extra=True, log_prefix='', target_samples=None):
    # try:
    opt_init, update, get_params = adam(lr)
    update = jax.jit(update, static_argnums = (3, 4))
    opt_state = opt_init(params_flat)
    losses = []
    tracker = {'eps': [], 'gamma': []}
    looper = tqdm(range(iters)) if info.run_cluster == 0 else range(iters)
    for i in looper:
        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        seeds = jax.random.randint(rng_key, (info.N,), 1, 1e6)
        params_flat = get_params(opt_state)
        if info.run_cluster == 0:
            tracker['eps'].append(collect_eps(params_flat, unflatten, trainable))
            tracker['gamma'].append(collect_gamma(params_flat, unflatten, trainable))

            wandb.log({f'{log_prefix}/eps': onp.array(tracker['eps'][-1])})
            wandb.log({f'{log_prefix}/gamma': onp.array(tracker['gamma'][-1])})
        
        grad, (loss, z) = grad_and_loss(seeds, params_flat, unflatten, params_fixed, log_prob_model)

        if "pretrain" not in log_prefix and i  % 100 == 0:
            if info.model == "nice":
                make_grid(z, info.im_size, n=64, wandb_prefix=f'{log_prefix}/images')
            if info.model == "many_gmm":
                plot_gmm(z, log_prob_model, info.loc_scaling, wandb_prefix=f'{log_prefix}/images')
            if target_samples is not None:
                if info.model == "nice":
                    make_grid(target_samples, info.im_size, n=64, wandb_prefix=f'{log_prefix}/target')
                wandb.log({f'{log_prefix}/w2': W2_distance(z, target_samples[:z.shape[0], ...])})

        losses.append(np.mean(loss).item())
        wandb.log({f'{log_prefix}/loss': np.mean(loss).item()})
        if np.isnan(np.mean(loss)):
            print('Diverged')
            return [], True, params_flat, tracker
        opt_state = update(i, grad, opt_state, unflatten, trainable)
    return losses, False, params_flat, tracker
    # except Exception as e:
    #     print('Sth failed!', e)
    #     print('Sth failed!', file = sys.stderr)
    #     print(e, file = sys.stderr)
    #     return [], True, None


def sample(info, n_samples, n_input_dist_seeds, params_flat, unflatten, params_fixed, log_prob_model, loss_fn, rng_key_gen, log_prefix=''):
    elbos, ln_zs = [], []
    zs = []

    eval_seeds = jax.random.randint(rng_key_gen, (n_samples * n_input_dist_seeds,), 1, 1e6)
    for i in range(n_input_dist_seeds):
        seeds = eval_seeds[i * n_samples : (i + 1) * n_samples]

        _, (loss_list, z) = loss_fn(seeds, params_flat, unflatten, params_fixed, log_prob_model)
        
        zs.append(z)
        elbos.append([x.item() for x in loss_list])
    
    zs = np.concatenate(zs, axis=0)

    return elbos, zs
