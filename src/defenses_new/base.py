from typing import Optional

import torch
from art.defences.preprocessor.preprocessor import PreprocessorPyTorch
from torch.autograd import Function
from torch.autograd.function import FunctionCtx


# noinspection PyMethodOverriding
class InstanceFunction(Function):
    """
    Custom function accepting a non-batched tensor.
    """

    @staticmethod
    def forward(ctx: FunctionCtx, x_t: torch.Tensor) -> torch.Tensor:
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

    def forward_one(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        Process one non-batched input tensor.

        :param x_t: Input tensor of shape (C, H, W)
        :return: Processed input.
        """
        raise NotImplementedError
