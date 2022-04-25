from random import sample

import torch
from art.defences.preprocessor.preprocessor import PreprocessorPyTorch

from src.defenses import InstancePreprocessorPyTorch


class Ensemble(InstancePreprocessorPyTorch):
    """
    A preprocessing defense that ensembles multiple preprocessors at random.
    """

    def __init__(self, preprocessors: list[InstancePreprocessorPyTorch], nb_samples: int | None = None):
        """
        A preprocessing defense that ensembles multiple preprocessors at random.

        :param preprocessors: List of candidate preprocessing defenses.
        :param nb_samples: Number of sampled preprocessors. Set to None to sample all preprocessors.
        """
        super().__init__()
        self.preprocessors = preprocessors
        self.nb_samples = nb_samples or len(preprocessors)

        if not 1 <= self.nb_samples <= (total := len(preprocessors)):
            raise ValueError(f'Cannot sample {nb_samples} from {total} preprocessors.')

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None):
        for preprocess in sample(self.preprocessors, self.nb_samples):
            x, y = preprocess.forward(x, y)
        return x, y

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        for preprocess in sample(self.preprocessors, self.nb_samples):
            x = preprocess.forward_one(x)
        return x

    def __repr__(self):
        fmt_string = f'{self.__class__.__name__}[k={self.nb_samples}](\n'
        fmt_string += ''.join(f'  {p},\n' for p in self.preprocessors)
        fmt_string += ')'
        return fmt_string
