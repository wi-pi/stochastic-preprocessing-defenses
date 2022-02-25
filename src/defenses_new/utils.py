import functools
from typing import Union

import numpy as np
import torch
from cachetools import cached

from src.defenses_new.base import InstanceFunction, InstancePreprocessorPyTorch
from src.utils.typing import FLOAT_INTERVAL, INT_INTERVAL


def rand(interval: Union[INT_INTERVAL, FLOAT_INTERVAL], size=None) -> np.ndarray:
    """
    Generate random number from a given interval.

    :param interval: Desired int or float interval, tuple of low (inclusive) and high (exclusive).
    :param size: NumPy size.
    :return: Generated random numbers in a NumPy array.
    """
    if not isinstance(interval, tuple):
        interval = (interval, interval)

    low, high = interval

    if isinstance(low, int):
        return np.random.randint(low, high, size=size)

    if isinstance(low, float):
        return np.random.uniform(low, high, size=size)

    raise NotImplementedError(f'Unknown interval {interval}.')


def rand_t(interval: Union[INT_INTERVAL, FLOAT_INTERVAL], size=None) -> torch.Tensor:
    """Same as rand, but returns as torch tensor.
    """
    out = rand(interval, size)
    return torch.from_numpy(out)


@cached(cache={}, key=lambda x, _: id(x))
def make_bpda_function(forward_one, self):
    """
    Create autograd Function class with identity BPDA.

    Identical `forward_one` methods share the same BPDA function (enforced by cache).

    :param forward_one: Defined forward method.
    :param self: Instance of InstancePreprocessorPyTorch.
    :return: BPDA Function.
    """

    class Func(InstanceFunction):

        @staticmethod
        def forward(ctx, x_t: torch.Tensor) -> torch.Tensor:
            return forward_one(self, x_t)

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
            return grad_output

    return Func


def bpda_identity(forward_one):
    """Decorator to add BPDA (identity) to the `forward_one` method.
    """

    @functools.wraps(forward_one)
    def wrapper(self: InstancePreprocessorPyTorch, x: torch.Tensor) -> torch.Tensor:
        bpda = make_bpda_function(forward_one, self)
        return bpda.apply(x)

    return wrapper
