import abc
from typing import Optional, Tuple

import torch
from art.defences.preprocessor.preprocessor import PreprocessorPyTorch


class Preprocessor(PreprocessorPyTorch, abc.ABC):
    """
    Base class for preprocessing defenses.
    """

    @abc.abstractmethod
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _estimate_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward(x)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self._forward(x), y

    def estimate_forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._estimate_forward(x)
