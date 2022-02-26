import functools
from typing import Optional

import torch
from art.defences.preprocessor.preprocessor import PreprocessorPyTorch
from cachetools import cached
from torch.autograd import Function
from torch.autograd.function import FunctionCtx


# noinspection PyMethodOverriding
class InstanceFunction(Function):
    """
    Custom function accepting a non-batched tensor.
    """

    @staticmethod
    def forward(ctx: FunctionCtx, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class InstancePreprocessorPyTorch(PreprocessorPyTorch):
    """
    A preprocessing defense that processes each instance independently in a batch.

    This is convenient for randomized defenses processing each sample with independent randomness.

    For differentiable preprocessors, they should:
      (1) Implement the `forward_one` method.

    For non-differentiable preprocessors, they should:
      (1) For identity BPDA, implement `forward_one` and decorate it with `bpda_identity`.
      (2) For complex BPDA, define `InstanceFunction` and apply it in `forward_one`.
    """

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        """
        Top-level interface for preprocessing defense.

        :param x: Input batch data tensor of shape (B, C, H, W)
        :param y: Input batch label tensor (optional).
        :return: Processed batch of data and label.
        """
        processed_x = torch.stack(list(map(self.forward_one, x)))
        return processed_x, y

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process one non-batched input tensor.

        :param x: Input tensor of shape (C, H, W)
        :return: Processed input.
        """
        raise NotImplementedError


@cached(cache={}, key=lambda x, _: id(x))
def _make_bpda_function(forward_one, self):
    """
    Create autograd Function class with identity BPDA.

    Identical `forward_one` methods share the same BPDA function (enforced by cache).

    :param forward_one: Defined forward method.
    :param self: Instance of InstancePreprocessorPyTorch.
    :return: BPDA Function.
    """

    class InstanceFunctionBPDA(InstanceFunction):

        @staticmethod
        def forward(ctx, x: torch.Tensor) -> torch.Tensor:
            return forward_one(self, x)

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
            return grad_output

    return InstanceFunctionBPDA


def bpda_identity(forward_one):
    """Decorator to add BPDA (identity) to the `forward_one` method.
    """

    @functools.wraps(forward_one)
    def wrapper(self: InstancePreprocessorPyTorch, x: torch.Tensor) -> torch.Tensor:
        bpda = _make_bpda_function(forward_one, self)
        return bpda.apply(x)

    return wrapper
