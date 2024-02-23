from typing import Callable, Optional

import chex
import haiku as hk
import jax
import jax.numpy as np
from jax.example_libraries.stax import (
    Dense,
    FanInSum,
    FanOut,
    Identity,
    Softplus,
    parallel,
    serial,
)


def initialize_embedding(rng, nbridges, emb_dim, factor=0.05):
    return jax.random.normal(rng, shape=(nbridges, emb_dim)) * factor


def initialize_score_network(x_dim, emb_dim, nbridges, rho_dim=0, nlayers=2):
    in_dim = x_dim + rho_dim + emb_dim

    layers = []
    for i in range(nlayers):
        layers.append(
            serial(
                FanOut(2), parallel(Identity, serial(Dense(in_dim), Softplus)), FanInSum
            )
        )

    layers.append(Dense(x_dim))

    init_fun_nn, apply_fun_nn = serial(*layers)

    def init_fun(rng, input_shape):
        params = {}
        output_shape, params_nn = init_fun_nn(rng, (in_dim,))
        params["nn"] = params_nn
        rng, _ = jax.random.split(rng)
        params["emb"] = initialize_embedding(rng, nbridges, emb_dim)
        params["factor_sn"] = np.array(0.0)
        return output_shape, params

    def apply_fun(params, inputs, i, **kwargs):
        # inputs has size (x_dim +  rho_dim)
        emb = params["emb"][i, :]  # (emb_dim,)
        input_all = np.concatenate([inputs, emb])
        return apply_fun_nn(params["nn"], input_all) * params["factor_sn"]  # (x_dim,)

    return init_fun, apply_fun


def initialize_pis_network(x_dim, fully_connected_units, rho_dim=0):
    in_dim = x_dim + rho_dim

    def forward_fn(x, t):
        pisnet = PISNet(x_dim, fully_connected_units)
        return pisnet(x, t)

    pis_network_forward = hk.without_apply_rng(hk.transform(forward_fn))

    def init_fn(seed: chex.PRNGKey, shape=None):
        # match score_networks methods head and tail
        samples = jax.random.normal(seed, shape=(in_dim,))

        return None, pis_network_forward.init(seed, samples, 0)

    return init_fn, pis_network_forward.apply


def initialize_pis_grad_network(x_dim, fully_connected_units, rho_dim=0):
    in_dim = x_dim + rho_dim

    def forward_fn(x, t):
        pisnet = PISGRADNet(x_dim, fully_connected_units)
        return pisnet(x, t)

    pis_network_forward = hk.without_apply_rng(hk.transform(forward_fn))

    def init_fn(seed: chex.PRNGKey, shape=None):
        # match score_networks methods head and tail
        samples = jax.random.normal(seed, shape=(in_dim,))

        return None, pis_network_forward.init(seed, samples, 0)

    return init_fn, pis_network_forward.apply


class PISNet(hk.Module):
    def __init__(self, dim: int, fully_connected_units, name="drift_net"):
        super().__init__(name=name)

        fully_connected_units = [64, 64]
        self.fully_connected_units = fully_connected_units
        self.n_layers = len(fully_connected_units)
        self.n_channels = fully_connected_units[0]
        self.activation_fn = gelu

        # For most PIS_GRAD experiments channels = 64
        self.channels = self.n_channels
        self.timestep_phase = hk.get_parameter(
            "timestep_phase", shape=[1, self.channels], init=np.zeros
        )

        # Exact time_step coefs used in PIS GRAD
        self.timestep_coeff = np.linspace(start=0.1, stop=100, num=self.channels)[None]

        # This implements the time embedding for the non grad part of the network
        self.time_coder_state = hk.Sequential(
            [
                hk.Linear(self.channels),
                self.activation_fn,
                hk.Linear(self.channels),
            ]
        )

        # Time embedding and state concatenated network NN(x, emb(t))
        # This differs to PIS_grad where they do NN(Wx + emb(t))
        self.state_time_net = hk.Sequential(
            [
                hk.Sequential([hk.Linear(x), self.activation_fn])
                for x in fully_connected_units
            ]
            + [LinearZero(dim)]
        )

        self.nn_clip = 1.0e4

    def get_pis_timestep_embedding(self, timesteps: np.array):
        """PIS based timestep embedding.

        Args:
          timesteps: timesteps to embed

        Returns:
          embedded timesteps
        """

        sin_embed_cond = np.sin((self.timestep_coeff * timesteps) + self.timestep_phase)
        cos_embed_cond = np.cos((self.timestep_coeff * timesteps) + self.timestep_phase)
        return np.concatenate([sin_embed_cond, cos_embed_cond], axis=-1)

    def __call__(self, input_x: np.ndarray, time) -> np.ndarray:
        """Evaluates (carries out a forward pass) the model at train/inference time.

        Args:
            input_x:  state to the network (N_dim + rho)
            time:  time  to the network (1)
        Returns:
            returns an ndarray of logits (n_dim)
        """

        time_array_emb = self.get_pis_timestep_embedding(time)
        t_net_1 = self.time_coder_state(time_array_emb)

        t_net_1 = np.squeeze(t_net_1)
        extended_input = np.concatenate((input_x, t_net_1))
        out_state = self.state_time_net(extended_input)

        out_state = np.clip(out_state, -self.nn_clip, self.nn_clip)

        return out_state


def gelu(x):
    """We use this in place of jax.nn.relu because the approximation used.

    Args:
      x: input

    Returns:
      GELU activation
    """
    return x * 0.5 * (1.0 + jax.scipy.special.erf(x / np.sqrt(2.0)))


class LinearZero(hk.Module):
    """Linear layer with zero init."""

    def __init__(self, output_size, alpha=1, name=None):
        super().__init__(name=name)
        self.alpha = alpha
        self.output_size = output_size

    def __call__(self, x):
        j, k = x.shape[-1], self.output_size
        w = hk.get_parameter("w", shape=[j, k], dtype=x.dtype, init=np.zeros)
        b = hk.get_parameter("b", shape=[k], dtype=x.dtype, init=np.zeros)

        return np.dot(x, w) + b


class PISGRADNet(hk.Module):
    """PIS Grad network. Other than detaching should mimic the PIS Grad network.

    We detach the ULA gradients treating them as just features leading to much
    more stable training than PIS Grad.

    Attributes:
      config: ConfigDict specifying model architecture
    """

    def __init__(
        self,
        fully_connected_units: list,
        dim: int,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        self.stop_grad = True
        # self.architecture_specs = architecture_specs
        self.fully_connected_units = [64, 64]
        self.n_layers = len(self.fully_connected_units)
        self.activation_fn = gelu
        self.deactivate_grad_network = False

        # For most PIS_GRAD experiments channels = 64
        self.channels = self.fully_connected_units[0]
        self.timestep_phase = hk.get_parameter(
            "timestep_phase", shape=[1, self.channels], init=np.zeros
        )

        # Exact time_step coefs used in PIS GRAD
        self.timestep_coeff = np.linspace(start=0.1, stop=100, num=self.channels)[None]

        # This implements the time embedding for the non grad part of the network
        self.time_coder_state = hk.Sequential(
            [
                hk.Linear(self.channels),
                self.activation_fn,
                hk.Linear(self.channels),
            ]
        )

        # This carries out the time embedding for the NN(t) * log grad target
        self.time_coder_grad = hk.Sequential(
            [hk.Linear(self.channels)]
            + [
                hk.Sequential([self.activation_fn, hk.Linear(self.channels)])
                for _ in range(self.n_layers)
            ]
            + [self.activation_fn, LinearConsInit(dim, 0)]
        )

        # Time embedding and state concatenated network NN(x, emb(t))
        # This differs to PIS_grad where they do NN(Wx + emb(t))
        self.state_time_net = hk.Sequential(
            [
                hk.Sequential([hk.Linear(x), self.activation_fn])
                for x in self.fully_connected_units
            ]
            + [LinearZero(dim)]
        )

        self.state_dim = dim
        self.dim = dim + 1
        self._grad_ln = hk.LayerNorm(-1, True, True)
        self.nn_clip = 1.0e4
        self.lgv_clip = 1.0e2

    def get_pis_timestep_embedding(self, timesteps: np.array):
        """PIS based timestep embedding.

        Args:
          timesteps: timesteps to embed

        Returns:
          embedded timesteps
        """

        sin_embed_cond = np.sin((self.timestep_coeff * timesteps) + self.timestep_phase)
        cos_embed_cond = np.cos((self.timestep_coeff * timesteps) + self.timestep_phase)
        return np.concatenate([sin_embed_cond, cos_embed_cond], axis=-1)

    def __call__(
        self,
        input_array: np.ndarray,
        time_array: np.ndarray,
        target: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        training: Optional[bool] = True,
    ) -> np.ndarray:
        """Evaluates (carries out a forward pass) the model at train/inference time.

        Args:
            input_array:  state to the network (N_points, N_dim)
            time_array:  time  to the network (N_points, 1)
            target: ln pi target for ULA based features
            training: if true evaluates the network in training phase else inference

        Returns:
            returns an ndarray of logits (N_points, n_dim)
        """

        time_array_emb = self.get_pis_timestep_embedding(time_array)

        grad_bool = self.stop_grad
        # Using score information as a feature
        grad = hk.grad(lambda _x: target(_x).sum())(input_array)
        grad = jax.lax.stop_gradient(grad) if grad_bool else grad
        grad = np.clip(grad, -self.lgv_clip, self.lgv_clip)

        t_net_1 = self.time_coder_state(time_array_emb)
        t_net_2 = self.time_coder_grad(time_array_emb)

        extended_input = np.concatenate((input_array, t_net_1), axis=-1)
        out_state = self.state_time_net(extended_input)

        out_state = np.clip(out_state, -self.nn_clip, self.nn_clip)

        if self.deactivate_gradient_network:
            out_state_p_grad = out_state
        else:
            out_state_p_grad = out_state + t_net_2 * grad
        return out_state_p_grad
