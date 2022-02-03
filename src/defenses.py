from typing import Callable

import torch
from art.preprocessing.expectation_over_transformation.pytorch import EoTPyTorch


class EOTDefense(EoTPyTorch):

    def __init__(self, nb_samples: int, transforms: Callable):
        super().__init__(nb_samples=nb_samples, clip_values=(0, 1))
        self.transforms = transforms

    def _transform(self, x: torch.Tensor, y: torch.Tensor, **kwargs):
        x = self.transforms(x)
        return torch.clamp(x, *self.clip_values), y
