import functools
import jax
import equinox as eqx


def append_normalization(layers, norm, **kwargs):
    match norm:
        case 'instance_norm':
            layers.append(eqx.nn.GroupNorm(groups=kwargs['out_channels'], channels=kwargs['out_channels'], eps=1e-6))
        case 'layer_norm':
            layers.append(eqx.nn.LayerNorm(shape=kwargs['shape']))
        case 'none':
            pass
        case _:
            raise ValueError(f'unknown norm: {norm}')
    return layers


def append_activation(layers, activation, **kwargs):
    match activation:
        case 'relu':
            layers.append(eqx.nn.Lambda(jax.nn.relu))
        case 'leaky_relu':
            layers.append(eqx.nn.Lambda(functools.partial(jax.nn.leaky_relu, negative_slope=0.2)))
        case 'sigmoid':
            layers.append(eqx.nn.Lambda(jax.nn.sigmoid))
        case 'tanh':
            layers.append(eqx.nn.Lambda(jax.nn.tanh))
        case 'none':
            pass
        case _:
            raise ValueError(f'unknown activation: {activation}')
    return layers
