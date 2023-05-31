import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from . import base


class ImageDecoder(eqx.nn.Sequential, base.Decoder):
    transition_shape: tuple

    def __init__(self, dense_partial, transition_partial, conv_transpose_partial, conv_partial, *args, out_shape, x, transition_shape, key, **kwargs):
        dense_key, transition_key, conv_transpose_key, conv_key = jax.random.split(key, 4)
        dense = dense_partial(x=x, key=dense_key)
        x = dense(x)
        self.transition_shape = transition_shape
        transition = transition_partial(in_size=x.shape[0], out_size=np.prod(self.transition_shape), key=transition_key)
        x = transition(x)
        dense_to_conv = eqx.nn.Lambda(lambda x: jnp.reshape(x, self.transition_shape))
        x = dense_to_conv(x)
        conv_transpose = conv_transpose_partial(x=x, key=conv_transpose_key)
        x = conv_transpose(x)
        conv = conv_partial(in_channels=x.shape[0], out_channels=out_shape[0], key=conv_key)
        x = conv(x)
        assert x.shape == out_shape
        super().__init__(layers=[dense, transition, dense_to_conv, conv_transpose, conv])

    @eqx.filter_jit
    def __call__(self, z, *, key=None):
        outs = {
            'x_hat_logits': super().__call__(z)
        }
        return outs