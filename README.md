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
python main.py --config.boundmode MCD_CAIS_sn --config.model funnel --config.N 300 --config.alpha 0.05 --config.emb_dim 48 --config.init_eps 0.1 -config.init_sigma 1 --config.iters 11000 --noconfig.pretrain_mfvi --config.train_vi --noconfig.train_eps --config.wandb.name "funnel replicate w/ cos_sq" --config.lr 0.01 --config.n_samples 2000 --config.eps_schedule cos_sq
```
[Old wandb experiment with paper numbers](https://wandb.ai/shreyaspadhy/cais/runs/kh9n0y3n/workspace?workspace=user-shreyaspadhy)
[Wandb experiment](https://wandb.ai/shreyaspadhy/final_cmcd/runs/wka879ae?workspace=user-shreyaspadhy)

The paper numbers differ in the following ways: (1) Uses Geffner's manual ADAM implementation.

#### LGCP Experiments

```
python main.py --config.boundmode MCD_CAIS_sn --config.model lgcp --config.N 20 --config.alpha 0.05 --config.emb_dim 20 --config.init_eps 0.00001 -config.init_sigma 1 --config.iters 37500 --config.pretrain_mfvi --config.train_vi --config.train_eps --config.wandb.name "lgcp replicate" --config.lr 0.0001 --config.n_samples 500 --config.mfvi_iters 20000
```
[Old wandb experiment with paper numbers](https://wandb.ai/shreyaspadhy/cais/runs/jemnkjp5/workspace?workspace=user-shreyaspadhy)
[Wandb experiment](https://wandb.ai/shreyaspadhy/final_cmcd/runs/325oa9q7?workspace=user-shreyaspadhy)

Differences from the paper experiments: (1) The new run is about 10min slower, would be nice to figure out the bottleneck, (2) 20000 steps of MFVI is enough, vs 150k from the paper.

#### 2-GMM Experiments

```
python main.py --config.boundmode MCD_CAIS_sn --config.model gmm --config.N 300 --config.alpha 0.05 --config.emb_dim 20 --config.init_eps 0.01 -config.init_sigma 1 --config.iters 11000 --noconfig.pretrain_mfvi --config.train_vi --noconfig.train_eps --config.wandb.name "gmm replicate" --config.lr 0.001 --config.n_samples 500
```
[Old wandb experiment with rebuttal paper numbers](https://wandb.ai/shreyaspadhy/cais/runs/h9nwksr4/workspace?workspace=user-shreyaspadhy)
[Wandb experiment](https://wandb.ai/shreyaspadhy/final_cmcd/runs/1otzopu0?workspace=user-shreyaspadhy)
[Original paper numbers](https://wandb.ai/shreyaspadhy/cais/sweeps/n2exqhfq?workspace=user-shreyaspadhy)

Differences: (1) The new run has better $\ln Z$ estimates overall.

#### 40-GMM Experiments

```bash
python main.py --config.model many_gmm --config.boundmode MCD_CAIS_var_sn --config.N 2000 --config.nbridges 256 --noconfig.pretrain_mfvi --config.init_sigma 15 --config.grad_clipping --config.init_eps 0.65 --config.emb_dim 130 --config.lr 0.005 --noconfig.train_eps --noconfig.train_vi --config.wandb.name "logvar 40gmm"
```

```bash
python main.py --config.model many_gmm --config.boundmode MCD_CAIS_sn --config.N 2000 --config.nbridges 256 --noconfig.pretrain_mfvi --config.init_sigma 15 --config.grad_clipping --config.init_eps 0.1 --config.emb_dim 130 --config.lr 0.005 --noconfig.train_eps --noconfig.train_vi --config.wandb.name "kl 40gmm"
```

[Old KL Wandb experiment eps=0.65](https://wandb.ai/shreyaspadhy/cais/runs/5z3rdxgh?workspace=user-shreyaspadhy)
[Old KL Wandb experiment eps=0.1](https://wandb.ai/shreyaspadhy/cais/runs/2rigzwcd?workspace=user-shreyaspadhy)
[Old logvar Wandb experiment eps=0.65](https://wandb.ai/shreyaspadhy/cais/runs/9o0ccmpv?workspace=user-shreyaspadhy)
[Old logvar Wandb experiment eps=0.1](https://wandb.ai/shreyaspadhy/cais/runs/236aqlcp?workspace=user-shreyaspadhy)


