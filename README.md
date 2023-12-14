# CAIS

To run methods


`python main.py --config.model log_ionosphere --config.boundmode MCD_ULA`

- ULA uses MCD_ULA
- MCD uses MCD_ULA_sn
- UHA uses UHA
- LDVI uses MCD_U_a-lp-sn
- CAIS uses MCD_CAIS_sn
- 2nd order CAIS uses MCD_CAIS_UHA_sn


#### Funnel Experiments

```
python main.py --config.boundmode MCD_CAIS_sn --config.model funnel --config.N 300 --config.alpha 0.05 --config.emb_dim 48 --config.init_eps 0.1 -config.init_sigma 1 --config.iters 11000 --noconfig.pretrain_mfvi --config.train_vi --noconfig.train_eps --config.wandb.name "funnel replicate w/ cos_sq" --config.lr 0.01 --config.n_samples 2000 --config.beta_schedule cos_sq
```
[Wandb experiment](https://wandb.ai/shreyaspadhy/final_cmcd/runs/wka879ae?workspace=user-shreyaspadhy)

