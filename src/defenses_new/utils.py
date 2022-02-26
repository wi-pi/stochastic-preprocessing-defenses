from typing import Union

import numpy as np
import torch

from src.utils.typing import FLOAT_INTERVAL, INT_INTERVAL


def rand(interval: Union[INT_INTERVAL, FLOAT_INTERVAL], size=None) -> Union[int, float, np.ndarray]:
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
