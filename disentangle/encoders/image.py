import jax
import jax.numpy as jnp
import equinox as eqx

import disentangle
from . import base


class ImageEncoder(base.Encoder):
    conv: eqx.Module
    conv_to_dense: eqx.Module
    dense: eqx.Module
    head: eqx.Module

    def __init__(self, conv_partial, dense_partial, *, x, out_size, key):
        conv_key, dense_key, head_key = jax.random.split(key, 3)
        self.conv = conv_partial(x=x, key=conv_key)
        x = self.conv(x)
        self.transition_shape = x.shape
        self.conv_to_dense = eqx.nn.Lambda(lambda x: jnp.reshape(x, -1))
        x = self.conv_to_dense(x)
        self.dense = dense_partial(x=x, key=dense_key)
        x = self.dense(x)
        self.head = eqx.nn.Linear(x.shape[0], out_size, key=head_key)

    def __call__(self, x, *, key=None):
        x = self.conv(x)
        x = self.conv_to_dense(x)
        features = self.dense(x)
        pre_z = self.head(features)
        outs = {
            'pre_z': pre_z,
            'features': features
        }
        return outs