import collections

import ipdb
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import typing

import disentangle
from . import ae


class VQVAE(ae.AE):

    @staticmethod
    def quantization_loss(continuous, quantized):
        return jnp.mean(jnp.square(jax.lax.stop_gradient(continuous) - quantized))

    @staticmethod
    def commitment_loss(continuous, quantized):
        return jnp.mean(jnp.square(continuous - jax.lax.stop_gradient(quantized)))

    @eqx.filter_jit
    def batched_loss(self, model, data, step, *args, key=None, **kwargs):
        outs = jax.vmap(model)(data['x'])
        quantization_loss = jax.vmap(self.quantization_loss)(outs['z_continuous'], outs['z_quantized'])
        commitment_loss = jax.vmap(self.commitment_loss)(outs['z_continuous'], outs['z_quantized'])
        binary_cross_entropy_loss = jax.vmap(disentangle.losses.binary_cross_entropy_loss)(outs['x_hat_logits'],
                                                                                           data['x'])
        partition_norms = self.partition_norms()

        loss = self.lambdas['binary_cross_entropy'] * binary_cross_entropy_loss + \
               self.lambdas['quantization'] * quantization_loss + \
               self.lambdas['commitment'] * commitment_loss

        metrics = {
            'loss': loss,
            'binary_cross_entropy_loss': binary_cross_entropy_loss,
            'quantization_loss': quantization_loss,
            'commitment_loss': commitment_loss,
        }
        metrics.update({
            f'params_{k_partition}/{k_norm}': v
            for k_norm, v_norm in partition_norms.items() for k_partition, v in v_norm.items()
        })

        aux = {
            'metrics': metrics,
            'outs': outs,
        }

        return jnp.mean(loss), aux

