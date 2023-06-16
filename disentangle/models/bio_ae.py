import jax
import jax.numpy as jnp
import equinox as eqx

import disentangle
from . import ae


class BioAE(ae.AE):

    @staticmethod
    def nonnegative_loss(z_hat):
        return jnp.mean(jnp.maximum(-z_hat, 0))

    @staticmethod
    def activity_loss(z_hat):
        return jnp.mean(jnp.square(z_hat))

    @eqx.filter_jit
    def batched_loss(self, model, data, step, *args, key=None, **kwargs):
        outs = jax.vmap(model)(data['x'])
        binary_cross_entropy_loss = jax.vmap(disentangle.losses.binary_cross_entropy_loss)(outs['x_hat_logits'], data['x'])
        nonnegative_loss = jax.vmap(self.nonnegative_loss)(outs['z_hat'])
        activity_loss = jax.vmap(self.activity_loss)(outs['z_hat'])
        loss = self.lambdas['binary_cross_entropy'] * binary_cross_entropy_loss + \
               self.lambdas['nonnegative'] * nonnegative_loss + \
               self.lambdas['activity'] * activity_loss

        metrics = {
            'loss': loss,
            'binary_cross_entropy_loss': binary_cross_entropy_loss,
            'nonnegative_loss': nonnegative_loss,
            'activity_loss': activity_loss,
        }

        aux = {
            'metrics': metrics,
            'outs': outs,
        }

        return jnp.mean(loss), aux
