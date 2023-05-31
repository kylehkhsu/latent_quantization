import collections
import typing

import jax
import jax.numpy as jnp
import equinox
import equinox as eqx
import optax
import ipdb

import disentangle
from . import info_gan


class QuantizedInfoGAN(info_gan.InfoGAN):
    
    @staticmethod
    def quantization_loss(continuous, quantized):
        return jnp.mean(jnp.square(jax.lax.stop_gradient(continuous) - quantized))

    @staticmethod
    def commitment_loss(continuous, quantized):
        return jnp.mean(jnp.square(continuous - jax.lax.stop_gradient(quantized)))
    
    @eqx.filter_jit
    def batched_generator_loss(self, model, data, *, key):
        batch_size = data['x'].shape[0]
        latent_key, _ = jax.random.split(key, 2)
        z_fake = jax.vmap(model.latent.sample)(key=jax.random.split(latent_key, batch_size))

        x_fake = jax.nn.sigmoid(jax.vmap(model.generator)(z_fake)['x_hat_logits'])
        outs_fake = jax.vmap(model.encode)(x_fake)

        latent_loss = jax.vmap(model.latent_loss)(outs_fake['z_hat'], z_fake)
        quantization_loss = jax.vmap(model.quantization_loss)(outs_fake['z_continuous'], outs_fake['z_quantized'])
        commitment_loss = jax.vmap(model.commitment_loss)(outs_fake['z_continuous'], outs_fake['z_quantized'])

        generator_loss = (
            model.lambdas['value'] * -outs_fake['value'] +
            model.lambdas['latent'] * latent_loss +
            model.lambdas['quantization'] * quantization_loss +
            model.lambdas['commitment'] * commitment_loss
        )

        encoder_partition_norms = self.partition_norms_attr(self.encoder, 'encoder')
        generator_partition_norms = self.partition_norms_attr(self.generator, 'generator')

        metrics = {
            'generator_loss': generator_loss,
            'latent_loss': latent_loss,
        }
        metrics.update({
            f'{name}_params_{k_partition}/{k_norm}': v
            for name, partition_norms in
            zip(['encoder', 'generator'], [encoder_partition_norms, generator_partition_norms]) for k_norm, v_norm in
            partition_norms.items() for k_partition, v in v_norm.items()
        })
        outs = {
            'z_fake': z_fake,
            'x_fake': x_fake
        }
        aux = {
            'metrics': metrics,
            'outs': outs
        }

        return jnp.mean(generator_loss), aux

    @eqx.filter_jit
    def batched_encoder_loss(self, model, data, *, key):
        batch_size = data['x'].shape[0]
        latent_key, epsilon_key = jax.random.split(key, 2)

        z_fake = jax.vmap(model.latent.sample)(key=jax.random.split(latent_key, batch_size))
        epsilon = jax.random.uniform(epsilon_key, (batch_size, 1, 1, 1))

        x_fake = jax.nn.sigmoid(jax.vmap(model.generator)(z_fake)['x_hat_logits'])
        x_interpolated = epsilon * data['x'] + (1 - epsilon) * x_fake

        outs_fake = jax.vmap(model.encode)(x_fake)
        outs_real = jax.vmap(model.encode)(data['x'])

        x_gradient = jax.vmap(jax.grad(model.encoder.value))(x_interpolated)
        x_gradient_norm = jax.vmap(jnp.linalg.norm)(x_gradient)
        gradient_loss = jnp.square(x_gradient_norm - 1)

        value_loss = outs_fake['value'] - outs_real['value']

        latent_loss = jax.vmap(model.latent_loss)(outs_fake['z_hat'], z_fake)
        quantization_loss = jax.vmap(model.quantization_loss)(outs_fake['z_continuous'], outs_fake['z_quantized'])
        commitment_loss = jax.vmap(model.commitment_loss)(outs_fake['z_continuous'], outs_fake['z_quantized'])

        encoder_loss = (
            model.lambdas['value'] * value_loss +
            model.lambdas['gradient'] * gradient_loss +
            model.lambdas['latent'] * latent_loss +
            model.lambdas['quantization'] * quantization_loss +
            model.lambdas['commitment'] * commitment_loss
        )

        metrics = {
            'encoder_loss': encoder_loss,
            'value_loss': value_loss,
            'gradient_loss': gradient_loss,
            'latent_loss': latent_loss,
            'quantization_loss': quantization_loss,
            'commitment_loss': commitment_loss,
        }

        outs = {}

        aux = {
            'metrics': metrics,
            'outs': outs
        }
        return jnp.mean(encoder_loss), aux

