import abc

import equinox as eqx


class Block(eqx.Module, abc.ABC):

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass
