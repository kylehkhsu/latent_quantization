import ipdb
import jax
import jax.numpy as jnp
import optax
import equinox as eqx


def binary_cross_entropy_loss(x_hat_logits, x_true_probs):
    return jnp.mean(optax.sigmoid_binary_cross_entropy(x_hat_logits, x_true_probs))
