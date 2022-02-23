from random import sample
from typing import List, Optional, Tuple

import torch

from src.defenses.base import RandomizedPreprocessor
from src.utils.registry import get_params


class Ensemble(RandomizedPreprocessor):
    """
    Ensemble of preprocessing defenses.
    """
    params = ['preprocessors', 'k']

    def __init__(self, randomized: bool, preprocessors: List[RandomizedPreprocessor], k: int = 0):
        """
        Initialize an ensemble of multiple preprocessors.

        :param randomized: Set True to enable randomized ensemble.
        :param preprocessors: Candidate preprocessors for ensemble.
        :param k: Number of preprocessors to sample, if randomized.
        """
        super().__init__(**get_params())
        self.all_preprocessors = preprocessors
        self.k = min(k, len(preprocessors))

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        stateful: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Top-level interface for non-differentiable inference.

        Here we simply call each preprocessor's forward.

        :param x: Batch of inputs [batch, channel, height, width]
        :param y: Batch of labels [batch, classes]
        :param stateful: Set True if this forward is stateful (w.r.t. subsequent BPDA pass).
        :return: Processed batch of inputs.
        """
        params = self._get_params(save=stateful)
        for preprocess in params['preprocessors']:
            x, _ = preprocess.forward(x, y=None, stateful=stateful)

        return x, y

    def estimate_forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        stateful: bool = False
    ) -> torch.Tensor:
        """
        Top-level interface for differentiable inference (i.e., BPDA).

        Here we simply call each preprocessor's estimate_forward.

        :param x: Batch of inputs [batch, channel, height, width]
        :param y: Batch of labels [batch, classes]
        :param stateful: Set True if this forward is stateful (w.r.t. subsequent BPDA pass).
        :return: Processed batch of inputs.
        """
        params = self._get_params(load=stateful)
        for preprocess in params['preprocessors']:
            x = preprocess.estimate_forward(x, stateful=stateful)
        return x

    # we directly overwrite the forward method, no need for this one
    def _forward_one(self, x: torch.Tensor, indices: List[int] = None) -> torch.Tensor:
        raise NotImplementedError

    # we directly overwrite the forward method, no need for this one
    def _estimate_forward_one(self, x: torch.Tensor, indices: List[int] = None) -> torch.Tensor:
        raise NotImplementedError

    def get_random_params(self) -> dict:
        params = {
            'preprocessors': sample(self.all_preprocessors, self.k)
        }
        return params

    def __repr__(self):
        fmt_string = f'{self.__class__.__name__}(\n'
        fmt_string += f'    randomized={self.randomized and self.k},\n'
        fmt_string += f'    preprocessors=[\n'
        fmt_string += ''.join(f'        {p},\n' for p in self.all_preprocessors)
        fmt_string += '    ]\n'
        fmt_string += ')'
        return fmt_string
