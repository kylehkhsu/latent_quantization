import ipdb
import typing
import jax
import jax.numpy as jnp
import equinox as eqx
import disentangle

from . import base


class StyleConvTranspose2DBlock(base.Block):
    forward_block: eqx.nn.Sequential
    conditioning: eqx.Module
    instance_norm: eqx.nn.GroupNorm

    def __init__(self, *args, w_size, key, **kwargs):
        forward_key, conditioning_key = jax.random.split(key, 2)
        self.forward_block = disentangle.blocks.ConvTranspose2DBlock(*args, key=forward_key, **kwargs)
        assert isinstance(self.forward_block.layers[0], eqx.nn.ConvTranspose2d)
        assert len(self.forward_block.layers) == 2      # no normalization
        num_channels = self.forward_block.layers[0].out_channels
        self.conditioning = disentangle.blocks.DenseBlock(w_size, 2 * num_channels, key=conditioning_key)

        self.instance_norm = eqx.nn.GroupNorm(groups=num_channels, eps=1e-6, channelwise_affine=False)

    def __call__(self, x, w, *, key=None):
        scale, bias = jnp.split(self.conditioning(w), 2, axis=0)
        x = self.forward_block(x)
        x = self.instance_norm(x)
        x = x * scale[:, None, None] + bias[:, None, None]
        return x


class StyleConvTranspose2DBlocks(base.Block):
    blocks: typing.Tuple[StyleConvTranspose2DBlock]

    def __init__(self, widths, kernel_sizes, strides, paddings, activation, norm, *, x, w_size, key):
        blocks = []
        keys = jax.random.split(key, len(widths))
        assert len(widths) == len(kernel_sizes) == len(strides) == len(paddings)
        for i, (width, kernel_size, stride, padding, key) in enumerate(
                zip(widths, kernel_sizes, strides, paddings, keys)
        ):
            block = StyleConvTranspose2DBlock(
                in_channels=x.shape[0] if i == 0 else widths[i - 1],
                out_channels=width,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                activation=activation,
                norm=norm,
                w_size=w_size,
                key=key
            )
            blocks.append(block)
        self.blocks = tuple(blocks)

    def __call__(self, x, w, *, key=None):
        for i, block in enumerate(self.blocks):
            x = block(x, w)
        return x

def _test():
    block = StyleConvTranspose2DBlock(
        in_channels=8,
        out_channels=16,
        kernel_size=3,
        stride=1,
        padding=1,
        w_size=12,
        key=jax.random.PRNGKey(0)
    )

    x = jnp.ones((8, 4, 4))
    w = jnp.ones((12,))

    block(x, w)

if __name__ == '__main__':
    _test()