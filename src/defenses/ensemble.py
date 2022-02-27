from random import sample
from typing import List, Optional

import torch
from art.defences.preprocessor.preprocessor import PreprocessorPyTorch

from src.defenses import InstancePreprocessorPyTorch


class Ensemble(InstancePreprocessorPyTorch):
    """
    A preprocessing defense that ensembles multiple preprocessors at random.
    """

    def __init__(self, preprocessors: List[PreprocessorPyTorch], k: Optional[int] = None):
        """
        A preprocessing defense that ensembles multiple preprocessors at random.

        :param preprocessors: List of candidate preprocessing defenses.
        :param k: Number of sampled preprocessors. Set to None to sample all preprocessors.
        """
        super().__init__()
        self.preprocessors = preprocessors
        self.k = k if k is not None else len(preprocessors)
        assert 1 <= self.k <= len(preprocessors), f'Cannot sample {k} from {len(preprocessors)} preprocessors.'

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        for preprocess in sample(self.preprocessors, self.k):
            x, y = preprocess.forward(x, y)
        return x, y

    def __repr__(self):
        fmt_string = f'{self.__class__.__name__}[k={self.k}](\n'
        fmt_string += ''.join(f'  {p},\n' for p in self.preprocessors)
        fmt_string += ')'
        return fmt_string
