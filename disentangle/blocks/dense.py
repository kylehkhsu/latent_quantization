import jax
import equinox as eqx

from . import utils, base


class DenseBlock(eqx.nn.Sequential, base.Block):
    def __init__(self, in_size, out_size, norm='none', activation='relu', use_bias=True, *, key):
        layers = []
        linear_key, _ = jax.random.split(key)
        layers.append(eqx.nn.Linear(in_size, out_size, use_bias=use_bias, key=linear_key))
        layers = utils.append_normalization(layers, norm, shape=(out_size,))
        layers = utils.append_activation(layers, activation)
        super().__init__(layers)


class DenseBlocks(eqx.nn.Sequential, base.Block):

    def __init__(self, widths, activation, norm, x, *, key):
        blocks = []
        keys = jax.random.split(key, len(widths))
        for i, (width, key) in enumerate(zip(widths, keys)):
            block = DenseBlock(
                in_size=x.shape[0] if i == 0 else widths[i - 1],
                out_size=widths[i],
                activation=activation,
                norm=norm,
                key=key
            )
            blocks.append(block)
        super().__init__(blocks)
