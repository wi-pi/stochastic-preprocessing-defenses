import functools
import types
from typing import Callable


def averaged_method(method: Callable, n_calls: int):
    """
    Make a method average its outputs over multiple calls.

    :param method: Bounded method of an instance.
    :param n_calls: Number of calls.
    :return: Averaged outputs.
    """

    assert isinstance(method, types.MethodType), f'Expected method type, but got {type(method)}.'
    assert n_calls >= 1, f'n_calls must be positive, but got {n_calls}.'

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        outputs = method(*args, **kwargs)
        for _ in range(n_calls - 1):
            outputs += method(*args, **kwargs)
        return outputs / n_calls

    return types.MethodType(wrapper, method.__self__)
