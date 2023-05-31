import typing
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

import disentangle.losses
from . import ae

import ipdb


class VAE(ae.AE):
    x_dim: int
    z_dim: int
    anneal: bool
    anneal_duration: int

    def __init__(self, anneal, anneal_duration, **kwargs):
        super().__init__(**kwargs)

        self.x_dim = int(np.prod(kwargs['x'].shape))
        self.z_dim = self.latent.num_latents

        self.anneal = anneal
        self.anneal_duration = anneal_duration

    def __call__(self, x, *, key):
        outs = self.encoder(x)
        outs.update(**self.latent(outs['pre_z'], key=key))
        if self.inference:
            outs.update(**self.decoder(outs['z_mu']))
        else:
            outs.update(**self.decoder(outs['z_sample']))
        return outs

    @staticmethod
    def linear_annealing(step, end_step, end_value):
        return jnp.minimum(end_value, end_value / end_step * step)

    @staticmethod
    def kl_loss(z_mu, z_sigma):
        """KL(q(z|x) || N(0, I))"""
        return jnp.mean(-jnp.log(z_sigma) + 0.5 * (z_mu ** 2 + z_sigma ** 2 - 1))

    @eqx.filter_jit
    def batched_loss(self, model, data, step, *, key):
        forward_key, _ = jax.random.split(key, 2)
        batch_size = data['x'].shape[0]
        outs = jax.vmap(model)(data['x'], key=jax.random.split(forward_key, batch_size))

        binary_cross_entropy_loss = jax.vmap(disentangle.losses.binary_cross_entropy_loss)(outs['x_hat_logits'], data['x'])
        kl_loss = jax.vmap(self.kl_loss)(outs['z_mu'], outs['z_sigma'])

        if self.anneal:
            beta = self.linear_annealing(step=step, end_step=self.anneal_duration, end_value=self.lambdas['kl'])
        else:
            beta = float(self.lambdas['kl'])

        loss = self.lambdas['binary_cross_entropy'] * binary_cross_entropy_loss + \
            (self.z_dim / self.x_dim) * (beta * kl_loss)    # undo mean in kl_loss, do mean in bce_loss

        metrics = {
            'loss': loss,
            'binary_cross_entropy_loss': binary_cross_entropy_loss,
            'kl_loss': kl_loss,
            'beta': jnp.array(beta)
        }

        aux = {
            'metrics': metrics,
            'outs': outs
        }
        return jnp.mean(loss), aux
