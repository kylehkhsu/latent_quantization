import jax
import jax.numpy as jnp
import equinox as eqx

import disentangle
from . import continuous


class NoisyClampedLatent(continuous.ContinuousLatent):
    clamp: float
    noise: float

    def __init__(self, clamp, noise, **kwargs):
        super().__init__(**kwargs)
        self.clamp = clamp
        self.noise = noise

    def __call__(self, x, *, key):
        x = jax.lax.clamp(-self.clamp, x, self.clamp)
        x = x + jax.random.normal(key, shape=x.shape) * self.noise
        outs = {
            'z_hat': x
        }
        return outs


