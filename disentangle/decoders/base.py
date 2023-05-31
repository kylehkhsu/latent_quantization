import abc

import equinox as eqx


class Decoder(eqx.Module, abc.ABC):

    @abc.abstractmethod
    def __call__(self, x, *, key=None):
        pass
