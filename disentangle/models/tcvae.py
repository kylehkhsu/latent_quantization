import typing
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

import disentangle.losses
from . import vae

import ipdb


class TCVAE(vae.VAE):
    dataset_size: int

    def __init__(self, dataset_size, **kwargs):
        super().__init__(**kwargs)

        self.dataset_size = dataset_size

    @eqx.filter_jit
    def batched_loss(self, model, data, step, *, key):
        forward_key, _ = jax.random.split(key, 2)
        batch_size = data['x'].shape[0]
        outs = jax.vmap(model)(data['x'], key=jax.random.split(forward_key, batch_size))

        log_p_z = jax.vmap(self.log_p_z)(outs['z_sample'])
        log_q_z_given_x = jax.vmap(self.log_q_z_given_x)(outs['z_sample'], outs['z_mu'], outs['z_sigma'])

        f_jk = jax.vmap(self.log_q_z_given_x_k, in_axes=(None, 0, 0), out_axes=0)
        f_ijk = jax.vmap(f_jk, in_axes=(0, None, None), out_axes=0)
        log_q_zi_given_xj_k = f_ijk(outs['z_sample'], outs['z_mu'], outs['z_sigma'])

        log_q_z = jax.scipy.special.logsumexp(jnp.sum(log_q_zi_given_xj_k, axis=2), axis=1) - jnp.log(batch_size) - jnp.log(self.dataset_size)
        log_prodk_q_zk = jnp.sum(
            jax.scipy.special.logsumexp(log_q_zi_given_xj_k, axis=1) - jnp.log(batch_size) - jnp.log(self.dataset_size),
            axis=1  # sum over k
        )

        mutual_information_loss = log_q_z_given_x - log_q_z
        total_correlation_loss = log_q_z - log_prodk_q_zk
        dimension_wise_kl_loss = log_prodk_q_zk - log_p_z

        binary_cross_entropy_loss = jax.vmap(disentangle.losses.binary_cross_entropy_loss)(outs['x_hat_logits'], data['x'])

        if self.anneal:
            total_correlation_lambda = self.linear_annealing(step=step, end_step=self.anneal_duration, end_value=self.lambdas['total_correlation'])
        else:
            total_correlation_lambda = float(self.lambdas['total_correlation'])

        loss = self.lambdas['binary_cross_entropy'] * binary_cross_entropy_loss + \
            (1 / self.x_dim) * (    # bce is reconstruction loss divided by x_dim
                self.lambdas['mutual_information'] * mutual_information_loss +
                total_correlation_lambda * total_correlation_loss +
                self.lambdas['dimension_wise_kl'] * dimension_wise_kl_loss
            )

        metrics = {
            'loss': loss,
            'binary_cross_entropy_loss': binary_cross_entropy_loss,
            'mutual_information_loss': mutual_information_loss,
            'total_correlation_loss': total_correlation_loss,
            'dimension_wise_kl_loss': dimension_wise_kl_loss,
            'total_correlation_lambda': jnp.array(total_correlation_lambda)
        }

        aux = {
            'metrics': metrics,
            'outs': outs
        }
        return jnp.mean(loss), aux

    @staticmethod
    def log_q_z_given_x(z, mu, sigma):
        return jax.scipy.stats.norm.logpdf(x=z, loc=mu, scale=sigma).sum()

    @staticmethod
    def log_q_z_given_x_k(z, mu, sigma):
        return jax.scipy.stats.norm.logpdf(x=z, loc=mu, scale=sigma)

    @staticmethod
    def log_p_z(z):
        return jax.scipy.stats.norm.logpdf(x=z, loc=0, scale=1).sum()
