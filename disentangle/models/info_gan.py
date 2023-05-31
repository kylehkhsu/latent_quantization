import collections
import typing

import jax
import jax.numpy as jnp
import equinox
import equinox as eqx
import optax
import ipdb

import disentangle
from . import base


class InfoGAN(base.Model):
    decoder = None
    generator: eqx.Module
    encoder: disentangle.encoders.CriticEncoder

    def __init__(self, latent_partial, encoder_partial, decoder_partial, lambdas, *, x, key, **kwargs):
        super().__init__()
        latent_key, encoder_key, critic_key, generator_key = jax.random.split(key, 4)
        in_shape = x.shape
        self.latent = latent_partial(key=latent_key)

        self.encoder = disentangle.encoders.CriticEncoder(
            encoder_partial(x=x, out_size=self.latent.num_inputs, key=encoder_key),
            x=x,
            key=critic_key
        )
        outs = self.encoder(x)
        outs.update(**self.latent(outs['pre_z']))

        self.generator = decoder_partial(
            x=outs['z_hat'], out_shape=in_shape, transition_shape=self.encoder.transition_shape, key=generator_key
        )
        outs.update(**self.generator(outs['z_hat']))

        self.lambdas = lambdas

        assert outs['x_hat_logits'].shape == in_shape

    @staticmethod
    def latent_loss(z_hat, z_fake):
        return jnp.mean(jnp.square(z_hat - z_fake))

    @eqx.filter_jit
    def batched_generator_loss(self, model, data, *, key):
        batch_size = data['x'].shape[0]
        latent_key, _ = jax.random.split(key, 2)
        z_fake = jax.vmap(model.latent.sample)(key=jax.random.split(latent_key, batch_size))

        x_fake = jax.nn.sigmoid(jax.vmap(model.generator)(z_fake)['x_hat_logits'])
        outs_fake = jax.vmap(model.encode)(x_fake)

        latent_loss = jax.vmap(model.latent_loss)(outs_fake['z_hat'], z_fake)

        generator_loss = (
                model.lambdas['value'] * -outs_fake['value'] +
                model.lambdas['latent'] * latent_loss
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
    def encode(self, x, *, key=None):
        outs = self.encoder(x)
        outs.update(**self.latent(outs['pre_z']))
        return outs

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

        encoder_loss = (
                model.lambdas['value'] * value_loss +
                model.lambdas['gradient'] * gradient_loss +
                model.lambdas['latent'] * latent_loss
        )

        metrics = {
            'encoder_loss': encoder_loss,
            'value_loss': value_loss,
            'gradient_loss': gradient_loss,
            'latent_loss': latent_loss,
        }

        outs = {}

        aux = {
            'metrics': metrics,
            'outs': outs
        }
        return jnp.mean(encoder_loss), aux

    def param_labels(self):
        param_labels = jax.tree_map(lambda _: 'regularized', self.filter())

        param_labels = disentangle.utils.relabel_attr(param_labels, 'bias', 'unregularized')
        param_labels = disentangle.utils.relabel_attr(param_labels, 'latent', 'unregularized')

        return param_labels

    def partition(self, module, labels):
        label_set = set(jax.tree_util.tree_leaves(labels))
        partition = {k: eqx.filter(self.filter(module),
                                   jax.tree_util.tree_map(lambda x: x == k, labels, is_leaf=lambda x: x in label_set))
                     for k in label_set}
        return partition

    @staticmethod
    def partition_norms(partition):
        ret = {}
        ret['l1'] = {k: jnp.sum(jnp.array([jnp.sum(jnp.abs(x)) for x in jax.tree_util.tree_leaves(v)])) for k, v in
                     partition.items()}
        ret['l2'] = {k: jnp.sqrt(jnp.sum(jnp.array([jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(v)]))) for
                     k, v in partition.items()}
        return ret

    def partition_norms_attr(self, attr, attr_name):
        return self.partition_norms(self.partition(attr, getattr(self.param_labels(), attr_name)))

    def construct_optimizer(self, config):
        param_labels = self.param_labels()
        optimizers = {
            'generator': optax.multi_transform(
                {
                    'regularized': optax.chain(
                        optax.clip(config.optim.clip),
                        optax.adamw(
                            config.optim.learning_rate,
                            config.optim.beta1,
                            config.optim.beta2,
                            weight_decay=config.model_partial.lambdas.l2
                        )
                    ),
                    'unregularized': optax.chain(
                        optax.adam(
                            config.optim.learning_rate,
                            config.optim.beta1,
                            config.optim.beta2,
                        )
                    )
                },
                param_labels=self.optax(param_labels.generator, param_labels.latent)
            ),
            'encoder': optax.multi_transform(
                {
                    'regularized': optax.chain(
                        optax.clip(config.optim.clip),
                        optax.adamw(
                            config.optim.learning_rate,
                            config.optim.beta1,
                            config.optim.beta2,
                            weight_decay=config.model_partial.lambdas.l2
                        )
                    ),
                    'unregularized': optax.chain(
                        optax.adam(
                            config.optim.learning_rate,
                            config.optim.beta1,
                            config.optim.beta2,
                        )
                    )
                },
                param_labels=disentangle.utils.optax(param_labels.encoder)
            )
        }
        optimizer_states = {
            'encoder': optimizers['encoder'].init(disentangle.utils.optax(self.filter().encoder)),
            'generator': optimizers['generator'].init(self.optax(self.filter().generator, self.filter().latent))
        }

        return optimizers, optimizer_states

    @staticmethod
    def optax(generator, latent):
        return {
            'generator': generator,
            'latent': latent
        }

    @staticmethod
    def unoptax(generator_and_latent):
        return generator_and_latent['generator'], generator_and_latent['latent']