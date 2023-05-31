import jax
import jax.numpy as jnp
import equinox as eqx

import disentangle
from . import base


class DiagonalGaussianLatent(base.Latent):
    def __init__(self, num_latents, *, key):
        self.is_continuous = True
        self.num_latents = num_latents
        self.num_inputs = 2 * num_latents

    def __call__(self, x, *, key):
        mu, sigma_param = jnp.split(x, 2)
        sigma = jax.nn.softplus(sigma_param)
        z_sample = jax.random.normal(key, shape=mu.shape) * sigma + mu
        outs = {
            'z_hat': mu,
            'z_sample': z_sample,
            'z_sigma': sigma,
            'z_mu': mu
        }
        return outs
