import jax
import jax.numpy as jnp


def peak_signal_to_noise_ratio(x_hat_logits, x_true):
    mse = jnp.mean(jnp.square(jax.nn.sigmoid(x_hat_logits) - x_true))
    max_value = 1.
    return 20 * jnp.log10(max_value) - 10 * jnp.log10(mse)

