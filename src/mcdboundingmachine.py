import jax
import jax.numpy as jnp
import mcd_utils
import variationaldist as vd
from jax.flatten_util import ravel_pytree
from nn import initialize_mcd_network


def initialize(
    dim,
    vdparams=None,
    nbridges=0,
    eps=0.01,
    gamma=10.0,
    eta=0.5,
    ngridb=32,
    mgridref_y=None,
    trainable=["eps"],
    use_score_nn=True,
    emb_dim=48,
    nlayers=3,
    seed=1,
    mode="MCD_U_lp-e",
):
    """
    Modes allowed:
        - MCD_ULA: This is ULA. Method from Thin et al.
        - MCD_ULA_sn: This is MCD. Method from Doucet et al.
        - MCD_U_a-lp: UHA but with approximate sampling of momentum (no score network).
        - MCD_U_a-lp-sn: Approximate sampling of momentum, followed by leapfrog, using score network(x, rho) for backward sampling.
        - MCD_CAIS_sn: CAIS with trainable SN.
        - MCD_CAIS_UHA_sn: CAIS underdampened with trainable SN.
    """
    params_train = {}  # Has all trainable parameters
    params_notrain = {}  # Non trainable parameters

    if "vd" in trainable:
        params_train["vd"] = vdparams
        if vdparams is None:
            params_train["vd"] = vd.initialize(dim)
    else:
        params_notrain["vd"] = vdparams
        if vdparams is None:
            params_notrain["vd"] = vd.initialize(dim)

    if "eps" in trainable:
        params_train["eps"] = eps
    else:
        params_notrain["eps"] = eps

    if "gamma" in trainable:
        params_train["gamma"] = gamma
    else:
        params_notrain["gamma"] = gamma

    if "eta" in trainable:
        params_train["eta"] = eta
    else:
        params_notrain["eta"] = eta

    # Initialise score networks if needed.
    if mode in [
        "MCD_ULA_sn",
        "MCD_U_e-lp-sna",
        "MCD_U_a-lp-sna",
        "MCD_CAIS_sn",
        "MCD_CAIS_var_sn",
    ]:
        init_fun_sn, apply_fun_sn = initialize_mcd_network(
            dim, emb_dim, nbridges, nlayers=nlayers
        )
        params_train["sn"] = init_fun_sn(jax.random.PRNGKey(seed), None)[1]
    elif mode in [
        "MCD_U_a-lp-sn",
        "MCD_U_ea-lp-sn",
        "MCD_U_a-nv-sn",
        "MCD_CAIS_UHA_sn",
    ]:
        # Initialise score networks with rho_dim also specified.
        init_fun_sn, apply_fun_sn = initialize_mcd_network(
            dim, emb_dim, nbridges, rho_dim=dim, nlayers=nlayers
        )
        params_train["sn"] = init_fun_sn(jax.random.PRNGKey(seed), None)[1]
    else:
        apply_fun_sn = None
        print("No score network needed by the method.")

    # Everything related to betas
    # betas are defined as a learnable function in [0, 1] given by normalised mgridref_y
    # betas = cumsum(mgridref_y) / sum(mgridref_y)
    if mgridref_y is not None:
        ngridb = mgridref_y.shape[0] - 1
    else:
        if nbridges < ngridb:
            ngridb = nbridges
        mgridref_y = jnp.ones(ngridb + 1) * 1.0
    params_notrain["gridref_x"] = jnp.linspace(0, 1, ngridb + 2)
    params_notrain["target_x"] = jnp.linspace(0, 1, nbridges + 2)[1:-1]
    if "mgridref_y" in trainable:
        params_train["mgridref_y"] = mgridref_y
    else:
        params_notrain["mgridref_y"] = mgridref_y

    # Other fixed parameters
    params_fixed = (dim, nbridges, mode, apply_fun_sn)
    params_flat, unflatten = ravel_pytree((params_train, params_notrain))
    return params_flat, unflatten, params_fixed


def compute_log_elbo(
    seed,
    params_flat,
    unflatten,
    params_fixed,
    log_prob,
    eps_schedule=None,
    grad_clipping=False,
):
    """
    Compute the log_ELBO as a sum/difference of log probabilities.

    The log_ELBO is defined as:
        L = log p(z_K) - log q(z_1) + \sum_{k=1}^{K-1} [log B_k(z_k  | z_{k+1}) - log F_k(z_k | z_{k+1})]
    """
    params_train, params_notrain = unflatten(params_flat)
    params_notrain = jax.lax.stop_gradient(params_notrain)
    params = {**params_train, **params_notrain}  # Gets all parameters in single place
    dim, nbridges, _, _ = params_fixed

    if nbridges >= 1:
        gridref_y = jnp.cumsum(params["mgridref_y"]) / jnp.sum(params["mgridref_y"])
        gridref_y = jnp.concatenate([jnp.array([0.0]), gridref_y])
        betas = jnp.interp(params["target_x"], params["gridref_x"], gridref_y)

    rng_key_gen = jax.random.PRNGKey(seed)

    rng_key, rng_key_gen = jax.random.split(rng_key_gen)

    # We sample z_1 ~ q(z), and L = - log q(z_1)
    z = vd.sample_rep(rng_key, params["vd"])
    w = -vd.log_prob(params["vd"], z)

    # Evolve UHA and update weight
    delta_H = jnp.array([0.0])
    if nbridges >= 1:
        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        # Evolve the trajectory to calculate z_K
        # w_mom = \sum_{k=1}^{K-1} [log B_k(z_k  | z_{k+1}) - log F_k(z_k | z_{k+1})]
        z, w_mom, _ = mcd_utils.evolve(
            z,
            betas,
            params,
            rng_key,
            params_fixed,
            log_prob,
            eps_schedule=eps_schedule,
            grad_clipping=grad_clipping,
        )
        w += w_mom

    # Add log p(z_K) to L
    w = w + log_prob(z)
    return -1.0 * w, (z, _)


# @functools.partial(jax.jit, static_argnums = (2, 3, 4))
def compute_bound(
    seeds,
    params_flat,
    unflatten,
    params_fixed,
    log_prob,
    eps_schedule=None,
    grad_clipping=False,
):
    # Vmap over a batch of samples (identified by seeds)
    batch_log_elbos, (z, _) = jax.vmap(
        compute_log_elbo, in_axes=(0, None, None, None, None, None, None)
    )(
        seeds,
        params_flat,
        unflatten,
        params_fixed,
        log_prob,
        eps_schedule,
        grad_clipping,
    )
    # batch_log_elbos, (z, _) = compute_log_elbo(seeds[0], params_flat, unflatten, params_fixed, log_prob)
    return batch_log_elbos.mean(), (batch_log_elbos, z)


def compute_bound_var(
    seeds,
    params_flat,
    unflatten,
    params_fixed,
    log_prob,
    eps_schedule=None,
    grad_clipping=False,
):
    batch_log_elbos, (z, _) = jax.vmap(
        compute_log_elbo, in_axes=(0, None, None, None, None, None, None)
    )(
        seeds,
        params_flat,
        unflatten,
        params_fixed,
        log_prob,
        eps_schedule,
        grad_clipping,
    )

    return jnp.clip(batch_log_elbos.var(ddof=0), -1e7, 1e7), (batch_log_elbos, z)
