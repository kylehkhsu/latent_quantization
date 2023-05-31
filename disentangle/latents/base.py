import abc

import equinox as eqx


class Latent(eqx.Module, abc.ABC):
    is_continuous: bool
    num_latents: int
    num_inputs: int

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass
