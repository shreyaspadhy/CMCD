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
from jax.config import config as jax_config
from model_handler import load_model
from utils import (
    calculate_W2_distances,
    flatten_nested_dict,
    log_final_losses,
    make_grid,
    setup_config,
    setup_training,
    update_config_dict,
)

jax_config.update("jax_traceback_filtering", "off")

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

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
#   - DNF uses MCD_DNF


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

        # If tractable distribution, we also return sample_from_target_fn
        if config.model in TRACTABLE_DISTS:
            log_prob_model, dim, sample_from_target_fn = load_model(
                config.model, config
            )
        else:
            log_prob_model, dim = load_model(config.model, config)
            sample_from_target_fn = None

        # Set up random seeds
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
            losses, params_flat, _ = opt.run(
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
                use_ema=False,
            )
            vdparams_init = unflatten(params_flat)[0]["vd"]

            elbo_init = -jnp.mean(jnp.array(losses[-500:]))
            print("Done training initial parameters, got ELBO %.2f." % elbo_init)
            wandb.log({"elbo_init": np.array(elbo_init)})

        if config.boundmode == "UHA":
            trainable = "eta"
            if config.train_eps:
                trainable = trainable + ("eps",)
            if config.train_vi:
                trainable = trainable + ("vd",)
            if config.train_betas:
                trainable = trainable + ("mgridref_y",)
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
            trainable = ("eta", "gamma")
            if config.train_eps:
                trainable = trainable + ("eps",)
            if config.train_vi:
                trainable = trainable + ("vd",)
            if config.train_betas:
                trainable = trainable + ("mgridref_y",)

            print(f"Params being trained : {trainable}")
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
                    eps_schedule=config.eps_schedule,
                    grad_clipping=config.grad_clipping,
                )
            else:
                compute_bound_fn = partial(
                    mcdbm.compute_bound,
                    eps_schedule=config.eps_schedule,
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

        _, params_flat, ema_params = opt.run(
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
            use_ema=config.use_ema,
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

        final_elbo, final_ln_Z = log_final_losses(eval_losses)

        print("Done training, got ELBO %.2f." % final_elbo)
        print("Done training, got ln Z %.2f." % final_ln_Z)

        if config.use_ema:
            eval_losses_ema, samples_ema = opt.sample(
                config,
                n_samples,
                n_input_dist_seeds,
                ema_params,
                unflatten,
                params_fixed,
                log_prob_model,
                loss_fn,
                eval_rng_key_gen,
                log_prefix="eval",
            )

            final_elbo_ema, final_ln_Z_ema = log_final_losses(
                eval_losses_ema, log_prefix="_ema"
            )

            print("With EMA, got ELBO %.2f." % final_elbo_ema)
            print("With EMA, got ln Z %.2f." % final_ln_Z_ema)

        # Plot samples
        if config.model in ["nice", "funnel", "gmm"]:
            other_target_samples = sample_from_target_fn(
                jax.random.PRNGKey(2), samples.shape[0]
            )

            calculate_W2_distances(
                samples,
                target_samples,
                other_target_samples,
                n_samples,
                config.n_input_dist_seeds,
                n_samples,
            )

            if config.use_ema:
                calculate_W2_distances(
                    samples_ema,
                    target_samples,
                    other_target_samples,
                    n_samples,
                    config.n_input_dist_seeds,
                    n_samples,
                    log_prefix="_ema",
                )

            if config.model == "nice":
                make_grid(
                    samples, config.im_size, n=64, wandb_prefix="images/final_sample"
                )
                if config.use_ema:
                    make_grid(
                        samples_ema,
                        config.im_size,
                        n=64,
                        wandb_prefix="images/final_sample_ema",
                    )

        params_train, params_notrain = unflatten(params_flat)
        params = {**params_train, **params_notrain}

        if config.wandb.log_artifact:
            artifact_name = f"{config.model}_{config.boundmode}_{config.nbridges}"

            artifact = wandb.Artifact(
                artifact_name,
                type="final params",
            )

            # Save model
            with artifact.new_file("params.pkl", "wb") as f:
                pickle.dump(params, f)

            wandb.log_artifact(artifact)


if __name__ == "__main__":
    os.environ["WANDB_API_KEY"] = "9835d6db89010f73306f92bb9a080c9751b25d28"

    # Adds jax flags to the program.
    jax.config.config_with_absl()

    # Snippet to pass configs into the main function.
    def _main(argv):
        del argv
        config = FLAGS.config
        main(config)

    app.run(_main)
