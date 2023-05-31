import typing

import jax
import jax.numpy as jnp
import equinox as eqx

import disentangle
from . import base


class VectorQuantizedLatent(base.Latent):
    num_embeddings: int
    embedding_size: int
    embeddings: jnp.array

    def __init__(self, num_latents, num_embeddings, embedding_size, key):
        values_key, _ = jax.random.split(key, 2)
        self.is_continuous = False
        self.num_latents = num_latents
        self.num_inputs = num_latents * embedding_size

        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        s = 1.0 / self.embedding_size
        limit = jnp.sqrt(3.0 * s)
        self.embeddings = jax.random.uniform(key, (num_embeddings, embedding_size), minval=-limit, maxval=limit)

    def quantize(self, x):
        def squared_distance(e):
            return jnp.sum(jnp.square(x - e))
        squared_distances = jax.vmap(squared_distance)(self.embeddings)
        index = jnp.argmin(squared_distances)
        return self.embeddings[index], index

    def __call__(self, x, *, key=None):
        x_ = x.reshape((self.num_latents, self.embedding_size))
        quantized, indices = jax.vmap(self.quantize)(x_)
        quantized = quantized.reshape((-1,))
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
