import typing
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

import disentangle


def relabel_attr(pytree, attr, label):
    has_attr = lambda x: hasattr(x, attr)
    where_attr = lambda m: tuple(
        getattr(x, attr) for x in jax.tree_util.tree_leaves(m, is_leaf=has_attr) if has_attr(x)
    )
    pytree = eqx.tree_at(where_attr, pytree, replace_fn=lambda _: label, is_leaf=lambda x: x is None)
    return pytree


def transpose_and_gather(auxs):
    """Return a tree where each leaf is the gather of the corresponding leaf in each tree in auxs."""
    aux = jax.tree_util.tree_transpose(
        jax.tree_util.tree_structure([0 for _ in auxs]),
        jax.tree_util.tree_structure(auxs[0]),
        auxs
    )
    aux = jax.tree_map(
        lambda x: jnp.concatenate(x) if x[0].ndim >= 1 else jnp.stack(x),
        aux,
        is_leaf=lambda x: isinstance(x, list) and (isinstance(x[0], jnp.ndarray) or isinstance(x[0], np.ndarray))
    )
    return aux


# list trick: https://github.com/patrick-kidger/equinox/issues/79
def optax(model: disentangle.models.Model):
    return [model]


def unoptax(model: typing.List[disentangle.models.Model]):
    return model[0]
