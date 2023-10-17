import os
import pickle
from functools import partial

import boundingmachine as bm
import jax
import jax.numpy as jnp
import mcdboundingmachine as mcdbm
import ml_collections.config_flags
import numpy as np
import opt
import wandb
from absl import app, flags
from configs.base import TRACTABLE_DISTS
from jax import scipy as jscipy
from jax.config import config as jax_config
from model_handler import load_model
from utils import (
    W2_distance,
    flatten_nested_dict,
    make_grid,
    setup_config,
    setup_training,
    update_config_dict,
)

jax_config.update("jax_traceback_filtering", "off")


ml_collections.config_flags.DEFINE_config_file(
    "config",
    "configs/base.py",
    "Training configuration.",
    lock_config=False,
)
FLAGS = flags.FLAGS

# python main.py --config.model many_gmm --config.boundmode MCD_CAIS_var_sn --config.N 300 --config.nbridges 128 --noconfig.pretrain_mfvi --config.init_sigma 10 --config.grad_clipping --config.init_eps 0.65 --config.emb_dim 40  --noconfig.train_eps --noconfig.train_vi

# Boundmodes
#   - ULA uses MCD_ULA
#   - MCD uses MCD_ULA_sn
#   - UHA uses UHA
#   - LDVI uses MCD_U_a-lp-sn
#   - CAIS uses MCD_CAIS_sn
#   - CAIS_UHA uses MCD_CAIS_UHA_sn
#   - CAIS_var uses MCD_CAIS_var_sn


def main(config):
    wandb_kwargs = {
        "project": config.wandb.project,
        "entity": config.wandb.entity,
        "config": flatten_nested_dict(config.to_dict()),
        "name": config.wandb.name if config.wandb.name else None,
        "mode": "online" if config.wandb.log else "disabled",
        "settings": wandb.Settings(code_dir=config.wandb.code_dir),
    }
    with wandb.init(**wandb_kwargs) as run:
        setup_training(run)
        # Load in the correct LR from sweeps
        new_vals = setup_config(run.config, config)
        update_config_dict(config, run, new_vals)

        print(config)

        if config.model in TRACTABLE_DISTS:
            log_prob_model, dim, sample_from_target_fn = load_model(config.model, config)
        else:
            log_prob_model, dim = load_model(config.model, config)
            sample_from_target_fn = None
        rng_key_gen = jax.random.PRNGKey(config.seed)

        train_rng_key_gen, eval_rng_key_gen = jax.random.split(rng_key_gen)

        # Train initial variational distribution to maximize the ELBO
        trainable = ("vd",)
        params_flat, unflatten, params_fixed = bm.initialize(
            dim=dim, nbridges=0, trainable=trainable, init_sigma=config.init_sigma
        )

        grad_and_loss = jax.jit(
            jax.grad(bm.compute_bound, 1, has_aux=True), static_argnums=(2, 3, 4)
        )
        if not config.pretrain_mfvi:
            mfvi_iters = 1
            vdparams_init = unflatten(params_flat)[0]["vd"]
        else:
            mfvi_iters = config.mfvi_iters
            losses, _, params_flat, _ = opt.run(
                config,
                config.mfvi_lr,
                mfvi_iters,
                params_flat,
                unflatten,
                params_fixed,
                log_prob_model,
                grad_and_loss,
                trainable,
                train_rng_key_gen,
                log_prefix="pretrain",
            )
            vdparams_init = unflatten(params_flat)[0]["vd"]

            elbo_init = -jnp.mean(jnp.array(losses[-500:]))
            print("Done training initial parameters, got ELBO %.2f." % elbo_init)
            wandb.log({"elbo_init": np.array(elbo_init)})

        if config.boundmode == "UHA":
            trainable = ("eta", "mgridref_y")
            if config.train_eps:
                trainable = trainable + ("eps",)
            if config.train_vi:
                trainable = trainable + ("vd",)
            params_flat, unflatten, params_fixed = bm.initialize(
                dim=dim,
                nbridges=config.nbridges,
                eta=config.init_eta,
                eps=config.init_eps,
                lfsteps=config.lfsteps,
                vdparams=vdparams_init,
                trainable=trainable,
            )
            grad_and_loss = jax.jit(
                jax.grad(bm.compute_bound, 1, has_aux=True), static_argnums=(2, 3, 4)
            )

            loss_fn = jax.jit(bm.compute_bound, static_argnums=(2, 3, 4))

        elif "MCD" in config.boundmode:
            trainable = ("eta", "gamma", "mgridref_y")
            if config.train_eps:
                trainable = trainable + ("eps",)
            if config.train_vi:
                trainable = trainable + ("vd",)

            print(trainable)
            params_flat, unflatten, params_fixed = mcdbm.initialize(
                dim=dim,
                nbridges=config.nbridges,
                vdparams=vdparams_init,
                eta=config.init_eta,
                eps=config.init_eps,
                trainable=trainable,
                mode=config.boundmode,
                emb_dim=config.emb_dim,
                nlayers=config.nlayers,
            )

            if "var" in config.boundmode:
                compute_bound_fn = partial(
                    mcdbm.compute_bound_var,
                    beta_schedule=config.beta_schedule,
                    grad_clipping=config.grad_clipping,
                )
            else:
                compute_bound_fn = partial(
                    mcdbm.compute_bound,
                    beta_schedule=config.beta_schedule,
                    grad_clipping=config.grad_clipping,
                )

            grad_and_loss = jax.jit(
                jax.grad(compute_bound_fn, 1, has_aux=True), static_argnums=(2, 3, 4)
            )
            loss_fn = jax.jit(compute_bound_fn, static_argnums=(2, 3, 4))

        else:
            raise NotImplementedError("Mode %s not implemented." % config.boundmode)

        # Average over 30 seeds, 500 samples each after training is done.
        n_samples = config.n_samples
        n_input_dist_seeds = config.n_input_dist_seeds

        if sample_from_target_fn is not None:
            target_samples = sample_from_target_fn(
                jax.random.PRNGKey(1), n_samples * n_input_dist_seeds
            )
        else:
            target_samples = None

        losses, diverged, params_flat, tracker = opt.run(
            config,
            config.lr,
            config.iters,
            params_flat,
            unflatten,
            params_fixed,
            log_prob_model,
            grad_and_loss,
            trainable,
            train_rng_key_gen,
            log_prefix="train",
            target_samples=target_samples,
        )

        eval_losses, samples = opt.sample(
            config,
            n_samples,
            n_input_dist_seeds,
            params_flat,
            unflatten,
            params_fixed,
            log_prob_model,
            loss_fn,
            eval_rng_key_gen,
            log_prefix="eval",
        )

        # (n_input_dist_seeds, n_samples)
        eval_losses = jnp.array(eval_losses)

        # Calculate mean and std of ELBOs over 30 seeds
        final_elbos = -jnp.mean(eval_losses, axis=1)
        final_elbo = jnp.mean(final_elbos)
        final_elbo_std = jnp.std(final_elbos)

        # Calculate mean and std of log Zs over 30 seeds
        ln_numsamp = jnp.log(n_samples)

        final_ln_Zs = (
            jscipy.special.logsumexp(-jnp.array(eval_losses), axis=1) - ln_numsamp
        )

        final_ln_Z = jnp.mean(final_ln_Zs)
        final_ln_Z_std = jnp.std(final_ln_Zs)

        print("Done training, got ELBO %.2f." % final_elbo)
        print("Done training, got ln Z %.2f." % final_ln_Z)

        wandb.log(
            {
                "elbo_final": np.array(final_elbo),
                "final_ln_Z": np.array(final_ln_Z),
                "elbo_final_std": np.array(final_elbo_std),
                "final_ln_Z_std": np.array(final_ln_Z_std),
            }
        )

        # Plot samples
        if config.model in ["nice", "funnel", "gmm"]:
            other_target_samples = sample_from_target_fn(
                jax.random.PRNGKey(2), samples.shape[0]
            )

            w2_dists, self_w2_dists = [], []
            for i in range(n_input_dist_seeds):
                samples_i = samples[i * n_samples: (i + 1) * n_samples, ...]
                target_samples_i = target_samples[
                    i * n_samples: (i + 1) * n_samples, ...
                ]
                other_target_samples_i = other_target_samples[
                    i * n_samples: (i + 1) * n_samples, ...
                ]
                w2_dists.append(W2_distance(samples_i, target_samples_i))
                self_w2_dists.append(
                    W2_distance(target_samples_i, other_target_samples_i)
                )

            if config.model == "nice":
                make_grid(samples, config.im_size, n=64, wandb_prefix="images/sample")

            wandb.log(
                {
                    "w2_dist": np.mean(np.array(w2_dists)),
                    "w2_dist_std": np.std(np.array(w2_dists)),
                    "self_w2_dist": np.mean(np.array(self_w2_dists)),
                    "self_w2_dist_std": np.std(np.array(self_w2_dists)),
                }
            )

        params_train, params_notrain = unflatten(params_flat)
        params = {**params_train, **params_notrain}

        if config.wandb.log_artifact:
            artifact_name = f"{config.model}_{config.boundmode}_{config.nbridges}"

            artifact = wandb.Artifact(
                artifact_name,
                type="nice_params",
                metadata={
                    **{
                        "alpha": config.alpha,
                        "n_bits": config.n_bits,
                        "im_size": config.im_size,
                    }
                },
            )

            # Save model
            with artifact.new_file("params.pkl", "wb") as f:
                pickle.dump(params, f)

            wandb.log_artifact(artifact)


if __name__ == "__main__":
    # if sys.argv:
    #     # pass wandb API as argv[1] and set environment variable
    #     # 'python mll_optim.py MY_API_KEY'
    os.environ["WANDB_API_KEY"] = "9835d6db89010f73306f92bb9a080c9751b25d28"

    # Adds jax flags to the program.
    jax.config.config_with_absl()

    # Snippet to pass configs into the main function.
    def _main(argv):
        del argv
        config = FLAGS.config
        main(config)

    app.run(_main)
