import abc
from queue import SimpleQueue
from typing import Optional, Tuple

import torch
from art.defences.preprocessor.preprocessor import PreprocessorPyTorch

from src.utils import Registry


class DEFENSES(object, metaclass=Registry):
    entries = dict()


class RandomizedPreprocessor(PreprocessorPyTorch, abc.ABC):
    """
    Base class for randomized preprocessing defenses.
    """

    def __init__(self, randomized: bool, **params):
        super().__init__()
        self.randomized = randomized
        self.realized_params = SimpleQueue()
        if not randomized:
            self.fixed_params = params

    @classmethod
    def as_fixed(cls, **params):
        """
        Realize a deterministic defense.

        :param params: Deterministic parameters.
        :return: A deterministic defense.
        """
        return cls(randomized=False, **params)

    @classmethod
    def as_randomized(cls, **params):
        """
        Realize a parameter-randomized defense.

        :param params: Other necessary parameters.
        :return: A parameter-randomized defense.
        """
        return cls(randomized=True, **params)

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        stateful: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Top-level interface for non-differentiable inference.

        For each sample, we generate a new set of parameters (if randomized) and apply the defense.

        :param x: Batch of inputs [batch, channel, height, width]
        :param y: Batch of labels [batch, classes]
        :param stateful: Set True if this forward is stateful (w.r.t. subsequent BPDA pass).
        :return: Processed batch of inputs.
        """
        x_processed = [None] * x.shape[0]
        for i in range(x.shape[0]):
            params = self._get_params(save=stateful)
            x_processed[i] = self._forward_one(x[i], **params).clip(0, 1)

        x_processed = torch.stack(x_processed).type_as(x)

        return x_processed, y

    def estimate_forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        stateful: bool = False
    ) -> torch.Tensor:
        """
        Top-level interface for differentiable inference (i.e., BPDA).

        For each sample, we retrieve the stored set of parameters (if randomized) and apply the defense.
        This ensures that non-BPDA and BPDA passes have the same parameter for the same sample.

        :param x: Batch of inputs [batch, channel, height, width]
        :param y: Batch of labels [batch, classes]
        :param stateful: Set True if this forward is stateful (w.r.t. previous non-BPDA pass).
        :return: Processed batch of inputs.
        """
        x_processed = [None] * x.shape[0]
        for i in range(x.shape[0]):
            params = self._get_params(load=stateful)
            x_processed[i] = self._estimate_forward_one(x[i], **params).clip(0, 1)

        x_processed = torch.stack(x_processed).type_as(x)

        return x_processed

    @abc.abstractmethod
    def _forward_one(self, x: torch.Tensor, **params) -> torch.Tensor:
        """
        Preprocess one sample with the given parameters (non-differentiable).

        :param x: One input [1, channel, height, width]
        :param params: A given set of parameters.
        :return: Processed input.
        """
        raise NotImplementedError

    def _estimate_forward_one(self, x: torch.Tensor, **params) -> torch.Tensor:
        """
        Preprocess one sample with the given parameters (BPDA).

        By default, we use the identity function as in BPDA.

        :param x: One input [1, channel, height, width]
        :param params: A given set of parameters.
        :return: Processed input.
        """
        return x

    def _get_params(self, save: bool = False, load: bool = False) -> dict:
        """
        Get parameters for the current inference.

        If not randomized, the fixed parameters are used.
        If randomized, either generate new or reuse the oldest parameters.

        :param save: Save generated parameters for later? (set True in non-BPDA pass)
        :param load: Load previously generated parameters? (set True in BPDA pass)
        :return: Parameters for this defense.
        """
        if save and load:
            raise ValueError('Cannot save and load at once.')

        if not self.randomized:
            return self.fixed_params

        if load:
            return self.realized_params.get_nowait()

        params = self.get_random_params()
        if save:
            self.realized_params.put_nowait(params)

        return params

    @abc.abstractmethod
    def get_random_params(self) -> dict:
        """
        Generate a random set of parameters for this defense.

        :return: Parameters for this defense.
        """
        raise NotImplementedError

    def __repr__(self):
        if self.randomized:
            return f'{self.__class__.__name__}(randomized=True)'

        params = ', '.join([f'{k}={self.fixed_params[k]}' for k in self.params])
        return f'{self.__class__.__name__}({params})'
