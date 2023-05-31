import ipdb
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

import disentangle
from . import base


class AE(base.Model):

    def __init__(self, latent_partial, encoder_partial, decoder_partial, lambdas, *, x, key, **kwargs):
        super().__init__()
        latent_key, encoder_key, decoder_key = jax.random.split(key, 3)
        in_shape = x.shape

        self.latent = latent_partial(key=latent_key)

        self.encoder = encoder_partial(x=x, out_size=self.latent.num_inputs, key=encoder_key)
        outs = self.encoder(x)
        outs.update(**self.latent(outs['pre_z'], key=latent_key))
        self.decoder = decoder_partial(x=outs['z_hat'], out_shape=in_shape, transition_shape=self.encoder.transition_shape, key=decoder_key)
        outs.update(**self.decoder(outs['z_hat']))

        self.lambdas = lambdas

    def __call__(self, x, *, key=None):
        outs = self.encoder(x)
        outs.update(**self.latent(outs['pre_z']))
        outs.update(**self.decoder(outs['z_hat']))
        return outs

    @eqx.filter_jit
    def batched_loss(self, model, data, step, *, key=None):
        outs = jax.vmap(model)(data['x'])
        binary_cross_entropy_loss = jax.vmap(disentangle.losses.binary_cross_entropy_loss)(outs['x_hat_logits'], data['x'])
        loss = (
            self.lambdas['binary_cross_entropy'] * binary_cross_entropy_loss
        )

        metrics = {
            'loss': loss,
            'binary_cross_entropy_loss': binary_cross_entropy_loss,
        }

        aux = {
            'metrics': metrics,
            'outs': outs,
        }

        return jnp.mean(loss), aux

    def construct_optimizer(self, config):
        weight_decay = config.model_partial.lambdas.get('l2', 0.0)
        optimizer = optax.multi_transform(
            {
                'regularized': optax.chain(
                    optax.clip(config.optim.clip),
                    optax.adamw(
                        learning_rate=config.optim.learning_rate,
                        weight_decay=weight_decay
                    )
                ),
                'unregularized': optax.chain(
                    optax.adam(
                        learning_rate=config.optim.learning_rate,
                    )
                )
            },
            param_labels=disentangle.utils.optax(self.param_labels())
        )
        optimizer_state = optimizer.init(disentangle.utils.optax(self.filter()))
        return optimizer, optimizer_state

    def param_labels(self):
        param_labels = jax.tree_map(lambda _: 'regularized', self.filter())

        param_labels = disentangle.utils.relabel_attr(param_labels, 'bias', 'unregularized')
        param_labels = disentangle.utils.relabel_attr(param_labels, 'latent', 'unregularized')

        return param_labels

    def partition(self):
        param_labels = self.param_labels()
        label_set = set(jax.tree_util.tree_leaves(param_labels))
        partition = {k: eqx.filter(self.filter(), jax.tree_util.tree_map(lambda x: x == k, param_labels, is_leaf=lambda x: x in label_set)) for k in label_set}
        return partition

    def partition_norms(self):
        partition = self.partition()
        ret = {}
        ret['l1'] = {k: jnp.sum(jnp.array([jnp.sum(jnp.abs(x)) for x in jax.tree_util.tree_leaves(v)])) for k, v in partition.items()}
        ret['l2'] = {k: jnp.sqrt(jnp.sum(jnp.array([jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(v)]))) for k, v in partition.items()}
        return ret

