import jax
import equinox as eqx

from . import utils, base


class ConvTranspose2DBlock(eqx.nn.Sequential, base.Block):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, norm='instance_norm',
                 activation='relu', use_bias=False, *, key):
        layers = []
        conv_key, _ = jax.random.split(key)
        layers.append(
            eqx.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=padding, use_bias=use_bias,
                                   key=conv_key))
        layers = utils.append_normalization(layers, norm, out_channels=out_channels)
        layers = utils.append_activation(layers, activation)
        super().__init__(layers)


class ConvTranspose2DBlocks(eqx.nn.Sequential, base.Block):
    def __init__(self, widths, kernel_sizes, strides, paddings, activation, norm, x, *, key):
        blocks = []
        keys = jax.random.split(key, len(widths))
        assert len(widths) == len(kernel_sizes) == len(strides) == len(paddings)
        for i, (width, kernel_size, stride, padding, key) in enumerate(
                zip(widths, kernel_sizes, strides, paddings, keys)
        ):
            block = ConvTranspose2DBlock(
                in_channels=x.shape[0] if i == 0 else widths[i - 1],
                out_channels=widths[i],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                activation=activation,
                norm=norm,
                key=key
            )
            blocks.append(block)
        super().__init__(blocks)
