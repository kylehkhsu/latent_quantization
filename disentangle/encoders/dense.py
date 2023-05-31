import jax
import jax.numpy as jnp
import equinox as eqx

import disentangle
from . import base


class DenseEncoder(base.Encoder):
    to_dense: eqx.Module
    dense: eqx.Module
    head: eqx.Module

    def __init__(self, dense_partial, *, x, out_size, key):
        conv_key, dense_key, head_key = jax.random.split(key, 3)
        self.transition_shape = x.shape
        self.to_dense = eqx.nn.Lambda(lambda x: jnp.reshape(x, -1))
        x = self.to_dense(x)
        self.dense = dense_partial(x=x, key=dense_key)
        x = self.dense(x)
        self.head = eqx.nn.Linear(x.shape[0], out_size, key=head_key)

    def __call__(self, x, *, key=None):
        x = self.to_dense(x)
        features = self.dense(x)
        pre_z = self.head(features)
        outs = {
            'pre_z': pre_z,
            'features': features
        }
        return outs