import abc
import collections
import typing

import jax
import equinox
import equinox as eqx
import optax
import ipdb

import disentangle


class Model(eqx.Module, abc.ABC):
    encoder: disentangle.encoders.Encoder
    latent: disentangle.latents.Latent
    decoder: disentangle.decoders.Decoder
    lambdas: typing.Mapping[str, float]
    inference: bool

    def __init__(self):
        self.inference = False

    def train(self):
        return eqx.tree_inference(self, False)

    def eval(self):
        return eqx.tree_inference(self, True)

    def filter(self, x=None):
        if x is None:
            return eqx.filter(self, eqx.is_inexact_array)
        else:
            return eqx.filter(x, eqx.is_inexact_array)

    @abc.abstractmethod
    def construct_optimizer(self, config):
        pass

    @abc.abstractmethod
    def param_labels(self):
        pass

