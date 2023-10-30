import functools
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from jax.flatten_util import ravel_pytree
from tqdm import tqdm
from utils import plot_samples


def project(x, unflatten, trainable):
    x_train, x_notrain = unflatten(x)
    if "eps" in trainable:
        x_train["eps"] = jnp.clip(x_train["eps"], 0.0000001, 0.5)
    if "eta" in trainable:
        x_train["eta"] = jnp.clip(x_train["eta"], 0, 0.99)
    if "gamma" in trainable:
        x_train["gamma"] = jnp.clip(x_train["gamma"], 0.001, None)
    if "mgridref_y" in trainable:
        x_train["mgridref_y"] = jax.nn.relu(x_train["mgridref_y"] - 0.001) + 0.001
    return ravel_pytree((x_train, x_notrain))[0]


def create_optimizer(step_size, b1=0.9, b2=0.999, eps=1e-8, trainable=None):
    if trainable is None:
        trainable = {}
    clip = optax.clip(5.0)
    optimizer = optax.adam(learning_rate=step_size, b1=b1, b2=b2, eps=eps)

    optimizer = optax.chain(clip, optimizer)

    return optimizer


@functools.partial(jax.jit, static_argnums=(1, 2))
def collect_eps(params_flat, unflatten, trainable):
    if "eps" in trainable:
        return unflatten(params_flat)[0]["eps"]
    else:
        return unflatten(params_flat)[1]["eps"]
    # return 0.


@functools.partial(jax.jit, static_argnums=(1, 2))
def collect_gamma(params_flat, unflatten, trainable):
    if "gamma" in trainable:
        return unflatten(params_flat)[0]["gamma"]
    return 0.0


def run(
    info,
    lr,
    iters,
    params_flat,
    unflatten,
    params_fixed,
    log_prob_model,
    grad_and_loss,
    trainable,
    rng_key_gen,
    extra=True,
    log_prefix="",
    target_samples=None,
):
    # opt_init, update, get_params = adam(lr)
    optimizer = create_optimizer(lr, trainable=trainable)

    opt_state = optimizer.init(params_flat)
    ema_params = deepcopy(params_flat)
    # train_step = jax.jit(train_step, static_argnums=(3,))
    # update = jax.jit(update, static_argnums=(3, 4))
    # opt_state = opt_init(params_flat)

    # Initialise EMA params
    # ema_params_flat = jax.tree_map(lambda x: x * 0.0, params_flat)
    # ema_step_size = info.ema_step_size

    losses = []
    tracker = {"eps": [], "gamma": []}
    looper = tqdm(range(iters)) if info.run_cluster == 0 else range(iters)

    for i in looper:
        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        seeds = jax.random.randint(rng_key, (info.N,), 1, 1e6)
        # params_flat = get_params(opt_state)

        grad, (loss, z) = grad_and_loss(
            seeds, params_flat, unflatten, params_fixed, log_prob_model
        )

        _, (ema_loss, z_ema) = grad_and_loss(
            seeds, ema_params, unflatten, params_fixed, log_prob_model
        )

        # Log samples every 1% of training steps.
        if "pretrain" not in log_prefix and i % (iters // 1000) == 0:
            plot_samples(
                info.model,
                log_prob_model,
                z,
                info,
                target_samples=target_samples,
                log_prefix=log_prefix,
                ema_samples=z_ema,
            )

            del z, z_ema

        if jnp.isnan(jnp.mean(loss)):
            print("Diverged")
            return [], True, params_flat, tracker

        updates, opt_state = optimizer.update(grad, opt_state, params_flat)
        params_flat = optax.apply_updates(params_flat, updates)
        params_flat = project(params_flat, unflatten, trainable)
        ema_params = optax.incremental_update(params_flat, ema_params, step_size=0.001)

        # opt_state = update(i, grad, opt_state, unflatten, trainable)
        # new_params_flat = get_params(opt_state)
        # ema_params_flat = jax.tree_map(
        #     lambda x, y: 0.999 * x + 0.001 * y, ema_params_flat, new_params_flat
        # )

        # Log scalar metrics every 0.1% of training steps.
        if i % (iters // 1000) == 0:
            # Convert to np array to prevent memory buildup after lots of training steps.
            loss, grad = np.array(loss), np.array(grad)
            ema_loss = np.array(ema_loss)
            log_dict = {
                f"{log_prefix}/loss": np.mean(loss),
                f"{log_prefix}/ema_loss": np.mean(ema_loss),
                f"{log_prefix}/grad": np.mean(grad),
                f"{log_prefix}/var_loss": np.var(loss, ddof=1),
                f"{log_prefix}/ema_var_loss": np.var(ema_loss, ddof=1),
                f"{log_prefix}/eps": np.array(
                    collect_eps(params_flat, unflatten, trainable)
                ),
                f"{log_prefix}/gamma": np.array(
                    collect_gamma(params_flat, unflatten, trainable)
                ),
                "train_step": i,
            }

            wandb.log(log_dict)
            losses.append(np.mean(loss))
    return losses, False, params_flat, tracker


def sample(
    info,
    n_samples,
    n_input_dist_seeds,
    params_flat,
    unflatten,
    params_fixed,
    log_prob_model,
    loss_fn,
    rng_key_gen,
    log_prefix="",
):
    elbos = []
    zs = []

    eval_seeds = jax.random.randint(
        rng_key_gen, (n_samples * n_input_dist_seeds,), 1, 1e6
    )
    for i in range(n_input_dist_seeds):
        seeds = eval_seeds[i * n_samples : (i + 1) * n_samples]

        _, (loss_list, z) = loss_fn(
            seeds, params_flat, unflatten, params_fixed, log_prob_model
        )

        zs.append(z)
        elbos.append([x.item() for x in loss_list])

    zs = jnp.concatenate(zs, axis=0)

    return elbos, zs
