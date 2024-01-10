import numpyro
import numpyro.distributions as dist
import jax.numpy as np


data = {
    "R": [10, 23, 23, 26, 17, 5, 53, 55, 32, 46, 10, 8, 10, 8, 23, 0, 3, 22, 15, 32, 3],
    "N": [
        39,
        62,
        81,
        51,
        39,
        6,
        74,
        72,
        51,
        79,
        13,
        16,
        30,
        28,
        45,
        4,
        12,
        41,
        30,
        51,
        7.0,
    ],
    "X1": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "X2": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    "tot": 21,
}

R = np.array(data["R"])
N = np.array(data["N"])
X1 = np.array(data["X1"])
X2 = np.array(data["X2"])
tot = data["tot"]


def load_model():
    def model(r):
        tau = numpyro.sample("tau", dist.Gamma(0.01, 0.01))
        a_0 = numpyro.sample("a_0", dist.Normal(0, 10))
        a_1 = numpyro.sample("a_1", dist.Normal(0, 10))
        a_2 = numpyro.sample("a_2", dist.Normal(0, 10))
        a_12 = numpyro.sample("a_12", dist.Normal(0, 10))
        with numpyro.plate("J", tot):
            b = numpyro.sample("b", dist.Normal(0, 1 / np.sqrt(tau)))
            logits = a_0 + a_1 * X1 + a_2 * X2 + a_12 * X1 * X2 + b
            r = numpyro.sample("r", dist.BinomialLogits(logits, N), obs=R)

    model_args = (R,)
    return model, model_args


# we expect the following
#
# var           mean       sd       median
# a_0           -.5525     .1852    -.5505
# a_1           0.08383    .3031    0.9076
# a_12          -.8165     .4109    -0.8073
# a_2           1.346      .2564    1.34
# 1/sqrt(tau)   .267       .1471    0.5929
