"""NICE TARGET
"""

from typing import Optional

import chex
import distrax
import haiku as hk
import jax
import jax.numpy as jnp

Array = jax.Array


class NICE(hk.Module):
    """Implements a NICE flow."""

    def __init__(
        self,
        dim: int,
        n_steps: int = 4,
        h_depth: int = 5,
        h_dim: int = 1000,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        self._dim = dim
        self._half_dim = dim // 2
        self._nets = []
        for _ in range(n_steps):
            layers = []
            for j in range(h_depth):
                if j != h_depth - 1:
                    layers.append(hk.Linear(h_dim))
                    layers.append(jax.nn.relu)
                else:
                    layers.append(hk.Linear(self._half_dim))
            net = hk.Sequential(layers)
            self._nets.append(net)

        self._parts = []
        self._inv_parts = []
        for _ in range(n_steps):
            shuff = list(reversed(range(dim)))
            self._parts.append(shuff)
            self._inv_parts.append(shuff)

        self._logscale = hk.get_parameter("logscale", (dim,), init=jnp.zeros)

    def forward(self, x: Array) -> Array:
        """Runs the model x->y."""
        chex.assert_shape(x, (None, self._dim))

        split = self._half_dim
        if self._dim % 2 == 1:
            split += 1

        for part, net in zip(self._parts, self._nets):
            x_shuff = x[:, part]
            xa, xb = x_shuff[:, :split], x_shuff[:, split:]
            ya = xa
            yb = xb + net(xa)
            x = jnp.concatenate([ya, yb], -1)

        chex.assert_shape(x, (None, self._dim))
        return x

    def reverse(self, y: Array) -> Array:
        """Runs the model y->x."""
        chex.assert_shape(y, (None, self._dim))

        split = self._half_dim
        if self._dim % 2 == 1:
            split += 1

        for inv_part, net in reversed(list(zip(self._inv_parts, self._nets))):
            ya, yb = y[:, :split], y[:, split:]
            xa = ya
            xb = yb - net(xa)
            x_shuff = jnp.concatenate([xa, xb], -1)
            y = x_shuff[:, inv_part]

        chex.assert_shape(y, (None, self._dim))
        return y

    def logpx(self, x: Array) -> Array:
        """Rreturns logp(x)."""
        z = self.forward(x)
        zs = z * jnp.exp(self._logscale)[None, :]

        pz = distrax.MultivariateNormalDiag(jnp.zeros_like(zs), jnp.ones_like(zs))
        logp = pz.log_prob(zs)
        logp = logp + self._logscale.sum()

        chex.assert_shape(logp, (x.shape[0],))
        return logp

    def sample(self, n: int) -> Array:
        """Draws n samples from model."""
        zs = jax.random.normal(hk.next_rng_key(), (n, self._dim))
        z = zs / jnp.exp(self._logscale)[None, :]
        x = self.reverse(z)

        chex.assert_shape(x, (n, self._dim))
        return x

    def reparameterized_sample(self, zs: Array) -> Array:
        """Draws n samples from model."""
        z = zs / jnp.exp(self._logscale)[None, :]
        x = self.reverse(z)

        chex.assert_shape(x, zs.shape)
        return x

    def loss(self, x: Array) -> Array:
        """Loss function for training."""
        return -self.logpx(x)
