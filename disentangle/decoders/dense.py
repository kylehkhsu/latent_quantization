import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from . import base


class DenseDecoder(eqx.nn.Sequential, base.Decoder):

    def __init__(self, dense_partial, *args, out_shape, x, key, **kwargs):
        dense_key, head_key = jax.random.split(key, 2)
        dense = dense_partial(x=x, key=dense_key)
        x = dense(x)
        head = eqx.nn.Linear(x.shape[0], np.prod(out_shape), key=key)
        x = head(x)
        to_output = eqx.nn.Lambda(lambda x: jnp.reshape(x, out_shape))
        x = to_output(x)
        assert x.shape == out_shape
        super().__init__(layers=[dense, head, to_output])

    @eqx.filter_jit
    def __call__(self, z, *, key=None):
        outs = {
            'x_hat_logits': super().__call__(z)
        }
        return outs