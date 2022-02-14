from typing import Sequence

import torch
import torch.nn as nn


class NormalizationLayer(nn.Module):
    """A normalization layer prepends a neural network.
    """

    PRESET = {
        'imagenet': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    }

    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        super().__init__()
        self.register_buffer('mean', torch.FloatTensor(mean)[..., None, None])
        self.register_buffer('std', torch.FloatTensor(std)[..., None, None])

    @classmethod
    def preset(cls, name: str):
        if name not in cls.PRESET:
            raise ValueError(f'Cannot find preset name "{name}".')
        mean, std = cls.PRESET[name]
        return cls(mean, std)

    def forward(self, x: torch.Tensor):
        if x.ndimension() != 4:
            raise ValueError(f'Only support a batch tensor of size (B, C, H, W), but got {x.size()}.')
        x = (x - self.mean) / self.std
        return x

    def __repr__(self):
        return f'NormalizationLayer(mean={self.mean}, std={self.std})'
