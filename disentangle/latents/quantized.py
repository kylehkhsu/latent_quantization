import typing

import jax
import jax.numpy as jnp
import equinox as eqx

import disentangle
from . import base


class QuantizedLatent(base.Latent):
    num_values_per_latent: list[int]
    _values_per_latent: list[jnp.ndarray]
    optimize_values: bool

    def __init__(self, num_latents, num_values_per_latent, optimize_values, key):
        values_key, _ = jax.random.split(key, 2)
        self.is_continuous = False
        self.num_latents = num_latents
        self.num_inputs = num_latents

        if isinstance(num_values_per_latent, int):
            self.num_values_per_latent = [num_values_per_latent] * num_latents
        else:
            self.num_values_per_latent = num_values_per_latent

        self._values_per_latent = [jnp.linspace(-0.5, 0.5, self.num_values_per_latent[i]) for i in range(num_latents)]
        self.optimize_values = optimize_values

    @property
    def values_per_latent(self):
        if self.optimize_values:
            return self._values_per_latent
        else:
            return [jax.lax.stop_gradient(v) for v in self._values_per_latent]

    @staticmethod
    def quantize(x, values):
        def distance(x, l):
            return jnp.abs(x - l)
        distances = jax.vmap(distance, in_axes=(None, 0))(x, values)
        index = jnp.argmin(distances)
        return values[index], index

    def __call__(self, x, *, key=None):
        quantized_and_indices = [self.quantize(x_i, values_i) for x_i, values_i in zip(x, self.values_per_latent)]
        quantized = jnp.stack([qi[0] for qi in quantized_and_indices])
        indices = jnp.stack([qi[1] for qi in quantized_and_indices])
        quantized_sg = x + jax.lax.stop_gradient(quantized - x)
        outs = {
            'z_continuous': x,
            'z_quantized': quantized,
            'z_hat': quantized_sg,
            'z_indices': indices
        }

        return outs

    def sample(self, *, key):
        ret = []
        for values, subkey in zip(self.values_per_latent, jax.random.split(key, self.num_latents)):
            ret.append(jax.random.choice(subkey, values))
        return jnp.array(ret)
