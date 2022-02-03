import random
from typing import Callable

import torch
import torchvision.transforms as T
from art.preprocessing.expectation_over_transformation.pytorch import EoTPyTorch


class EOT(EoTPyTorch):

    def __init__(self, transform: Callable, nb_samples: int):
        super().__init__(nb_samples=nb_samples, clip_values=(0, 1))
        self.transform = transform

    def _transform(self, x: torch.Tensor, y: torch.Tensor, **kwargs):
        x = self.transform(x)
        return torch.clamp(x, *self.clip_values), y


class RandomPickOne(T.Compose):

    def __call__(self, img):
        t = random.choice(self.transforms)
        return t(img)
