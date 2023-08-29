import datetime
import functools
import os
from typing import Sequence

from absl import app, flags
from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
from ml_collections import config_flags
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from nice import NICE


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", ("nice_config.py"),
                                "Run configuration.")


def dequantize(x, y, n_bits=3):
  n_bins = 2. ** n_bits
  x = tf.cast(x, tf.float32)
  x = tf.floor(x / 2 ** (8 - n_bits))
  x = x / n_bins
  x = x + tf.random.uniform(x.shape) / n_bins
  return x, y


def resize(x, y, im_size=28):
  """Resize images to desired size."""
  x = tf.image.resize(x, (im_size, im_size))
  return x, y


def logit(x, y, alpha=1e-6):
  """Scales inputs to rance [alpha, 1-alpha] then applies logit transform."""
  x = x * (1 - 2 * alpha) + alpha
  x = tf.math.log(x) - tf.math.log(1. - x)
  return x, y


def load_dataset(
    split: str,
    batch_size: int,
    im_size: int,
    alpha: float,
    n_bits: int
):
  """Loads the dataset as a generator of batches."""
  ds, ds_info = tfds.load("mnist", split=split,
                          as_supervised=True, with_info=True)
  ds = ds.cache()
  ds = ds.map(lambda x, y: resize(x, y, im_size=im_size),
              num_parallel_calls=tf.data.AUTOTUNE)
  ds = ds.map(lambda x, y: dequantize(x, y, n_bits=n_bits),
              num_parallel_calls=tf.data.AUTOTUNE)
  ds = ds.map(lambda x, y: logit(x, y, alpha=alpha),
              num_parallel_calls=tf.data.AUTOTUNE)
  ds = ds.shuffle(ds_info.splits["train"].num_examples)
  ds = ds.batch(batch_size, drop_remainder=True)
  ds = ds.prefetch(tf.data.AUTOTUNE)
  return tfds.as_numpy(ds)


def run_experiment(config):
  """Main experiment."""

  # flatboard
  # data_id = xdata.xm.get_data_id()
  # print("Set up flatboard, data_id %s", str(data_id))
  # writer_keys = ["train", "test"]
  # writers = {
  #     k: xdata.bt.writer(data_id, k)
  #     for k in writer_keys
  # }

  def forward_fn():
    flow = NICE(config.im_size ** 2, h_dim=config.hidden_dim)

    def _logpx(x):
      return flow.logpx(x)
    def _recons(x):
      return flow.reverse(flow.forward(x))
    return _logpx, (_logpx, _recons)

  forward = hk.multi_transform(forward_fn)
  rng_seq = hk.PRNGSequence(config.seed)

  # load data
  ds = load_dataset("train", config.batch_size, config.im_size, config.alpha,
                    config.n_bits)
  ds_test = load_dataset("test", config.batch_size, config.im_size,
                         config.alpha, config.n_bits)

  # get init data
  x, _ = next(iter(ds))
  x = x.reshape(x.shape[0], -1)

  params = forward.init(next(rng_seq), x)
  logpx_fn, recons_fn = forward.apply
  
  print("Param shapes:")
  print(jax.tree_map(lambda x: x.shape, params))

  x_re = recons_fn(params, next(rng_seq), x)
  print(f"Recons Error: {((x - x_re)**2).mean()}")

  opt = optax.adam(config.lr)
  opt_state = opt.init(params)

  # checkpointing
  # print("Setting up checkpointing.")
  # checkpoint = phoenix.Checkpoint(
  #     base_path=os.path.join(
  #         phoenix.checkpoint.get_default_path(),
  #         str(data_id.experiment_id),
  #         str(data_id.work_unit_id)),
  #     ttl=datetime.timedelta(days=120),
  #     history_length=20)
  
  iteration = 0
  data_mean = x.mean(0)
  data_std = x.std(0)
  # # checkpoint is for single device, params has device axis, use first
  # params = params
  # opt_state = opt_state
  # config = config
  # for (k, w) in writers.items():
  #   checkpoint.state.__setattr__(k, w)

  @functools.partial(jax.jit, static_argnums=3)
  def loss_fn(params, rng, x, with_wd=True):
    obj = logpx_fn(params, rng, x)

    l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
    if with_wd:
      return -obj.mean() + config.weight_decay * l2_loss
    else:
      return -obj

  @jax.jit
  def update(params, opt_state, rng, x):
    loss, grad = jax.value_and_grad(loss_fn)(params, rng, x)

    updates, opt_state = opt.update(grad, opt_state)
    new_params = optax.apply_updates(params, updates)
    return loss, new_params, opt_state

  for epoch in range(config.num_epochs):
    for x, _ in iter(ds):

      x = x.reshape(x.shape[0], -1)
      loss, params, opt_state = update(
          params, opt_state, next(rng_seq), x)
      
      # print(params)

      if iteration % config.log_interval == 0:
        print(f"Itr {iteration}, Epoch {epoch}, Loss {loss}")
        # checkpoint.state.__getattr__("train").write(
        #     {"itr": iteration,
        #      "loss": jax.device_get(loss)})

      # if iteration % config.save_interval == 0:
      #   checkpoint.save()

      iteration += 1

    test_loss = 0.
    n_seen = 0
    for x, _ in iter(ds_test):
      x = x.reshape(x.shape[0], -1)
      loss = loss_fn(params, next(rng_seq), x, with_wd=False)
      test_loss += loss.sum()
      n_seen += loss.shape[0]
    test_loss = test_loss / n_seen
    print(f"Epoch {epoch}, Test loss {test_loss}")
    # checkpoint.state.__getattr__("test").write({
    #     "itr": iteration,
    #     "loss": jax.device_get(test_loss)
    # })


def load_dataset(
    split: str,
    batch_size: int,
    im_size: int,
    alpha: float,
    n_bits: int
):
  """Loads the dataset as a generator of batches."""
  ds, ds_info = tfds.load("mnist", split=split,
                          as_supervised=True, with_info=True)
  ds = ds.cache()
  ds = ds.map(lambda x, y: resize(x, y, im_size=im_size),
              num_parallel_calls=tf.data.AUTOTUNE)
  ds = ds.map(lambda x, y: dequantize(x, y, n_bits=n_bits),
              num_parallel_calls=tf.data.AUTOTUNE)
  ds = ds.map(lambda x, y: logit(x, y, alpha=alpha),
              num_parallel_calls=tf.data.AUTOTUNE)
  ds = ds.shuffle(ds_info.splits["train"].num_examples)
  ds = ds.batch(batch_size, drop_remainder=True)
  ds = ds.prefetch(tf.data.AUTOTUNE)
  return tfds.as_numpy(ds)


def main(argv: Sequence[str]) -> None:
  config = FLAGS.config
  print("Displaying config %s", str(config))
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  run_experiment(config)


if __name__ == "__main__":
  print('test')
  app.run(main)