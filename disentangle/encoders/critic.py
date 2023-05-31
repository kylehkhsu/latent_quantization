import jax
import jax.numpy as jnp
import equinox as eqx

import disentangle
from . import base


class CriticEncoder(base.Encoder):
    base: base.Encoder
    critic: eqx.Module

    def __init__(self, encoder_base, *, x, key):
        critic_key, _ = jax.random.split(key, 2)

        self.base = encoder_base
        outs = self.base(x)

        self.critic = eqx.nn.Linear(outs['features'].shape[0], 1, key=critic_key)

    def __call__(self, x, *, key=None):
        outs = self.base(x)
        value = self.critic(outs['features'])
        outs['value'] = value
        return outs

    def value(self, x, *, key=None):
        outs = self.__call__(x)
        return jnp.squeeze(outs['value'])

    @property
    def transition_shape(self):
        return self.base.transition_shape
