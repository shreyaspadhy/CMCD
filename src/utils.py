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
from sinkhorn import sinkhorn
import torch

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



def W2_distance(x: Array, y: Array):

    # x, y is [n_samples, dim], [n_samples, dim]
    n_samples, dim = x.shape

    x, y = np.array(x).reshape((n_samples, dim)), np.array(y).reshape((n_samples, dim))

    # W2 = ot.sinkhorn2(a, b, M, reg=0.001)
    W2, _, _ = sinkhorn(torch.tensor(x), torch.tensor(y))

    # print(f'W2 : ', W2.shape)

    return W2


def sinkhorn_divergence(x: Array, y: Array, reg=1e-3):
    # x, y is [n_samples, dim], [n_samples, dim]
    n_samples, dim = x.shape

    x, y = np.array(x), np.array(y)
    a, b = np.ones((n_samples,)), np.ones((n_samples,))
    a, b = a / np.sum(a), b / np.sum(b)

    # W2 = np.sqrt(ot.emd2(a, b, ot.dist(x, y)))
    W2 = np.sqrt(ot.sinkhorn(a, b, ot.dist(x, y), reg=reg))

    return W2