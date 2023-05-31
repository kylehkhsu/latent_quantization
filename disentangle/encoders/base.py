import abc

import equinox as eqx


class Encoder(eqx.Module, abc.ABC):
    transition_shape: tuple

    @abc.abstractmethod
    def __call__(self, x, *, key=None):
        pass
