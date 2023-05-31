import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

import disentangle

from . import base


class StyleImageDecoder(base.Decoder):
    input_map: jnp.ndarray
    dense: disentangle.blocks.DenseBlocks
    style_conv_transpose: disentangle.blocks.StyleConvTranspose2DBlocks
    conv: disentangle.blocks.Conv2DBlock

    def __init__(self, dense_partial, style_conv_transpose_partial, conv_partial, input_channels, *args, transition_shape, out_shape, x, key, **kwargs):
        input_key, dense_key, style_conv_transpose_key, conv_key = jax.random.split(key, 4)
        input_shape = (input_channels,) + transition_shape[1:]
        self.input_map = 0.1 * jnp.ones(input_shape, jnp.float32)

        self.dense = dense_partial(x=x, key=dense_key)
        w = self.dense(x)

        self.style_conv_transpose = style_conv_transpose_partial(x=self.input_map, w_size=w.shape[0], key=style_conv_transpose_key)
        x = self.style_conv_transpose(x=self.input_map, w=w)

        self.conv = conv_partial(in_channels=x.shape[0], out_channels=out_shape[0], key=conv_key)
        x = self.conv(x)

        assert x.shape == out_shape

    @eqx.filter_jit
    def __call__(self, z, *, key=None):
        w = self.dense(z)
        outs = {
            'x_hat_logits': self.conv(self.style_conv_transpose(x=self.input_map, w=w))
        }
        return outs