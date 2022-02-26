from typing import Optional

import torch

from src.legacy.defenses.base import RandomizedPreprocessor


class EOT(RandomizedPreprocessor):
    """
    EOT Wrapper for randomized preprocessor.

    In non-stateful forward (i.e., no attack), EOT will invoke the preprocessor directly.
    In stateful forward (i.e., in attack), EOT will first repeat the inputs for several copies.
    """

    def __init__(self, preprocessor: RandomizedPreprocessor, nb_samples: int = 1):
        super().__init__(randomized=False)
        self.preprocessor = preprocessor
        self.nb_samples = nb_samples
        assert nb_samples >= 1

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None, stateful: bool = False):
        if stateful:
            x = self._repeat(x)
            y = self._repeat(y) if y is not None else y
        return self.preprocessor.forward(x, y, stateful)

    def estimate_forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None, stateful: bool = False):
        if stateful:
            x = self._repeat(x)
            y = self._repeat(y) if y is not None else y
        return self.preprocessor.estimate_forward(x, y, stateful)

    def _repeat(self, x: torch.Tensor) -> torch.Tensor:
        return torch.repeat_interleave(x, repeats=self.nb_samples, dim=0)

    def __repr__(self):
        return f'{self.__class__.__name__}(preprocessor={self.preprocessor}, nb_samples={self.nb_samples})'

    def _forward_one(self, x: torch.Tensor, **params) -> torch.Tensor:
        raise NotImplementedError

    def get_random_params(self) -> dict:
        raise NotImplementedError
