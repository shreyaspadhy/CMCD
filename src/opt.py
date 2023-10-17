import functools

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from jax.flatten_util import ravel_pytree
from tqdm import tqdm
from utils import plot_samples


def adam(step_size, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
    # Basically JAX's thing with added projection for some parameters.
    # Assumes ravel_pytree will always work the same way, so no need to update the
    # unflatten function (which may be problematic for jitting stuff)
    def init(x0):
        m0 = jnp.zeros_like(x0)
        v0 = jnp.zeros_like(x0)
        return x0, m0, v0

    def update(i, g, state, unflatten, trainable):
        def project(x, unflatten, trainable):
            x_train, x_notrain = unflatten(x)
            if 'eps' in trainable:
                x_train['eps'] = jnp.clip(x_train['eps'], 0.0000001, 0.5)
            if 'eta' in trainable:
                x_train['eta'] = jnp.clip(x_train['eta'], 0, 0.99)
            if 'gamma' in trainable:
                x_train['gamma'] = jnp.clip(x_train['gamma'], 0.001, None)
            if 'mgridref_y' in trainable:
                x_train['mgridref_y'] = jax.nn.relu(x_train['mgridref_y'] - 0.001) + 0.001
            return ravel_pytree((x_train, x_notrain))[0]
        x, m, v = state
        m = (1 - b1) * g + b1 * m  # First moment estimate
        v = (1 - b2) * jnp.square(g) + b2 * v  # Second moment estimate
        mhat = m / (1 - jnp.asarray(b1, m.dtype) ** (i + 1))  # Bias correction
        vhat = v / (1 - jnp.asarray(b2, m.dtype) ** (i + 1))
        x = x - step_size * mhat / (jnp.sqrt(vhat) + eps)
        x = project(x, unflatten, trainable)
        return x, m, v

    def get_params(state):
        x, _, _ = state
        return x
    return init, update, get_params


@functools.partial(jax.jit, static_argnums=(1, 2))
def collect_eps(params_flat, unflatten, trainable):
    if 'eps' in trainable:
        return unflatten(params_flat)[0]['eps']
    else:
        return unflatten(params_flat)[1]['eps']
    # return 0.


@functools.partial(jax.jit, static_argnums=(1, 2))
def collect_gamma(params_flat, unflatten, trainable):
    if 'gamma' in trainable:
        return unflatten(params_flat)[0]['gamma']
    return 0.


def run(info, lr, iters, params_flat, unflatten, params_fixed, log_prob_model, grad_and_loss, trainable, rng_key_gen,
        extra=True, log_prefix='', target_samples=None):

    opt_init, update, get_params = adam(lr)
    update = jax.jit(update, static_argnums=(3, 4))
    opt_state = opt_init(params_flat)
    losses = []
    tracker = {'eps': [], 'gamma': []}
    looper = tqdm(range(iters)) if info.run_cluster == 0 else range(iters)

    for i in looper:
        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        seeds = jax.random.randint(rng_key, (info.N,), 1, 1e6)
        params_flat = get_params(opt_state)

        grad, (loss, z) = grad_and_loss(seeds, params_flat, unflatten, params_fixed, log_prob_model)

        # Log samples every 1% of training steps.
        if "pretrain" not in log_prefix and i % (iters // 100) == 0:
            plot_samples(info.model, log_prob_model, z, info, target_samples=target_samples, log_prefix=log_prefix)
            del z
        if jnp.isnan(jnp.mean(loss)):
            print('Diverged')
            return [], True, params_flat, tracker
        opt_state = update(i, grad, opt_state, unflatten, trainable)

        # Log scalar metrics every 0.1% of training steps.
        if i % (iters // 1000) == 0:
            # Convert to np array to prevent memory buildup after lots of training steps.
            loss, grad = np.array(loss), np.array(grad)
            log_dict = {f'{log_prefix}/loss': np.mean(loss),
                        f'{log_prefix}/grad': np.mean(grad),
                        f'{log_prefix}/var_loss': np.var(loss, ddof=1),
                        f'{log_prefix}/eps': np.array(collect_eps(params_flat, unflatten, trainable)),
                        f'{log_prefix}/gamma': np.array(collect_gamma(params_flat, unflatten, trainable)),
                        'train_step': i}

            wandb.log(log_dict)
            losses.append(np.mean(loss))
    return losses, False, params_flat, tracker


def sample(info, n_samples, n_input_dist_seeds, params_flat, unflatten, params_fixed, log_prob_model, loss_fn,
           rng_key_gen, log_prefix=''):
    elbos = []
    zs = []

    eval_seeds = jax.random.randint(rng_key_gen, (n_samples * n_input_dist_seeds,), 1, 1e6)
    for i in range(n_input_dist_seeds):
        seeds = eval_seeds[i * n_samples: (i + 1) * n_samples]

        _, (loss_list, z) = loss_fn(seeds, params_flat, unflatten, params_fixed, log_prob_model)

        zs.append(z)
        elbos.append([x.item() for x in loss_list])

    zs = jnp.concatenate(zs, axis=0)

    return elbos, zs
