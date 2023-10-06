import os
from collections.abc import MutableMapping
from typing import List, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
import ml_collections
import wandb
from chex import Array
import numpy as np
import matplotlib.pyplot as plt
import ot
import itertools
import wandb


def make_grid(x: Array, im_size, n=16, wandb_prefix: str=""):
    x = np.array(x[:n].reshape(-1, im_size, im_size))

    n_rows = int(np.sqrt(n))
    fig, ax = plt.subplots(n_rows, n_rows, figsize=(8, 8))

    # Plot each image
    for i in range(n_rows):
        for j in range(n_rows):
            ax[i, j].imshow(x[i * n_rows + j], cmap='gray')
            ax[i, j].axis('off')
    
    # Log into wandb
    wandb.log({f"{wandb_prefix}": fig})
    plt.close()


def plot_contours_2D(log_prob_func,
                     ax: Optional[plt.Axes] = None,
                     bound=3, levels=20):
    """Plot the contours of a 2D log prob function."""
    if ax is None:
        fig, ax = plt.subplots(1)
    n_points = 200
    x_points_dim1 = np.linspace(-bound, bound, n_points)
    x_points_dim2 = np.linspace(-bound, bound, n_points)
    x_points = np.array(list(itertools.product(x_points_dim1, x_points_dim2)))
    log_probs = log_prob_func(x_points)
    log_probs = jnp.clip(log_probs, a_min=-1000, a_max=None)
    x1 = x_points[:, 0].reshape(n_points, n_points)
    x2 = x_points[:, 1].reshape(n_points, n_points)
    z = log_probs.reshape(n_points, n_points)
    ax.contour(x1, x2, z, levels=levels)


def plot_marginal_pair(samples,
                  ax = None,
                  marginal_dims = (0, 1),
                  bounds = (-5, 5),
                  alpha: float = 0.5):
    """Plot samples from marginal of distribution for a given pair of dimensions."""
    if not ax:
        fig, ax = plt.subplots(1)
    samples = jnp.clip(samples, bounds[0], bounds[1])
    ax.plot(samples[:, marginal_dims[0]], samples[:, marginal_dims[1]], "o", alpha=alpha)


def plot_gmm(samples, log_p_fn, loc_scaling, wandb_prefix: str=""):
    plot_bound = loc_scaling * 1.5
    fig, axs = plt.subplots(1, figsize=(5, 5))
    plot_marginal_pair(samples, axs, bounds=(-plot_bound, plot_bound))
    # plot_marginal_pair(x_smc, axs[1], bounds=(-plot_bound, plot_bound))
    # plot_marginal_pair(x_smc_resampled, axs[2], bounds=(-plot_bound, plot_bound))
    plot_contours_2D(log_p_fn, axs, bound=plot_bound, levels=50)
    # plot_contours_2D(log_p_fn, axs[1], bound=plot_bound, levels=50)
    # plot_contours_2D(log_p_fn, axs[2], bound=plot_bound, levels=50)
    axs.set_title("flow samples")
    # axs[1].set_title("smc samples")
    # axs[2].set_title("resampled smc samples")
    plt.tight_layout()
    wandb.log({f"{wandb_prefix}": wandb.Image(fig)})
    plt.close()
    # plt.show()
    # return plot


# Taken from https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
def flatten_nested_dict(nested_dict, parent_key="", sep="."):
    items = []
    for name, cfg in nested_dict.items():
        new_key = parent_key + sep + name if parent_key else name
        if isinstance(cfg, MutableMapping):
            items.extend(flatten_nested_dict(cfg, new_key, sep=sep).items())
        else:
            items.append((new_key, cfg))

    return dict(items)

def update_config_dict(config_dict: ml_collections.ConfigDict, run, new_vals: dict):
    config_dict.unlock()
    config_dict.update_from_flattened_dict(run.config)
    config_dict.update_from_flattened_dict(new_vals)
    run.config.update(new_vals, allow_val_change=True)
    config_dict.lock()


def setup_training(wandb_run):
    """Helper function that sets up training configs and logs to wandb."""
    if not wandb_run.config.get('use_tpu', False):
        # # TF can hog GPU memory, so we hide the GPU device from it.
        # tf.config.experimental.set_visible_devices([], "GPU")

        # Without this, JAX is automatically using 90% GPU for pre-allocation.
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"
        # os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"
        # Disable logging of compiles.
        jax.config.update("jax_log_compiles", False)

        # Log various JAX configs to wandb, and locally.
        wandb_run.summary.update(
            {
                "jax_process_index": jax.process_index(),
                "jax.process_count": jax.process_count(),
            }
        )
    else:
        # config.FLAGS.jax_xla_backend = "tpu_driver"
        # config.FLAGS.jax_backend_target = os.environ['TPU_NAME']
        # DEVICE_COUNT = len(jax.local_devices())
        print(jax.default_backend())
        print(jax.device_count(), jax.local_device_count())
        print("8 cores of TPU ( Local devices in Jax ):")
        print("\n".join(map(str, jax.local_devices())))


def W2_distance(x, y, reg = 0.01):
    N = x.shape[0]
    x, y = np.array(x), np.array(y)
    a,b = np.ones(N) / N, np.ones(N) / N

    M = ot.dist(x, y)
    M /= M.max()

    T_reg = ot.sinkhorn2(
        a, b, M, reg, log=False,
        numItermax=10000, stopThr=1e-16
    )
    return T_reg
