import ipdb
import jax
import equinox as eqx

import disentangle
from . import base


class ContinuousLatent(base.Latent):

    def __init__(self, num_latents, *, key):
        self.is_continuous = True
        self.num_latents = num_latents
        self.num_inputs = num_latents

    def __call__(self, x, *, key=None):
        outs = {
            'z_hat': x
        }
        return outs


class StandardGaussianLatent(ContinuousLatent):
    def sample(self, *, key):
        return jax.random.normal(key, (self.num_latents,))


class UniformLatent(ContinuousLatent):
    def sample(self, *, key):
        return jax.random.uniform(key, (self.num_latents,))