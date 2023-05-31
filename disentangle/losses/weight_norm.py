import ipdb
import jax
import jax.numpy as jnp

import equinox as eqx


def get_model_layers_with_name(model, name):
    has_name = lambda x: hasattr(x, name)
    layers, _ = jax.tree_util.tree_flatten(eqx.filter(model, has_name, is_leaf=has_name), is_leaf=has_name)
    return layers


def mean_squared_weight_norm(model):
    sum_of_square = 0
    num = 0
    for name in ['weight', 'bias']:
        layers = get_model_layers_with_name(model, name)
        for layer in layers:
            array = getattr(layer, name)
            if not isinstance(array, jnp.ndarray):
                continue
            sum_of_square += jnp.sum(jnp.square(array))
            num += jnp.prod(jnp.array(array.shape))
    return sum_of_square / num


def mean_absolute_weight_norm(model):
    sum_of_abs = 0
    num = 0
    for name in ['weight', 'bias']:
        layers = get_model_layers_with_name(model, name)
        for layer in layers:
            array = getattr(layer, name)
            if not isinstance(array, jnp.ndarray):
                continue
            sum_of_abs += jnp.sum(jnp.abs(array))
            num += jnp.prod(jnp.array(array.shape))
    return sum_of_abs / num
