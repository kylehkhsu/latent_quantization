import typing

import ipdb
import jax
import jax.numpy as jnp
import equinox as eqx

import disentangle
from . import base


class GlobalQuantizedLatent(base.Latent):
    num_values: int
    values: jnp.array

    def __init__(self, num_latents, num_values, key):
        values_key, _ = jax.random.split(key, 2)
        self.is_continuous = False
        self.num_latents = num_latents
        self.num_inputs = num_latents
        self.num_values = num_values
        self.values = jnp.linspace(-0.5, 0.5, self.num_values)

    @staticmethod
    def quantize(x, values):
        def distance(x, l):
            return jnp.abs(x - l)
        distances = jax.vmap(distance, in_axes=(None, 0))(x, values)
        index = jnp.argmin(distances)
        return values[index], index

    def __call__(self, x, *, key=None):
        quantized, indices = jax.vmap(self.quantize, in_axes=(0, None))(x, self.values)
        quantized_sg = x + jax.lax.stop_gradient(quantized - x)
        outs = {
            'z_continuous': x,
            'z_quantized': quantized,
            'z_hat': quantized_sg,
            'z_indices': indices
        }

        return outs

