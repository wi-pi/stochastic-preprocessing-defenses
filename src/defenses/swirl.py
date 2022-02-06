from typing import Tuple

import numpy as np
import torch
from skimage import transform

from src.defenses.base import DEFENSES, RandomizedPreprocessor
from src.utils import get_params


@DEFENSES
class Swirl(RandomizedPreprocessor):
    # FIXME: the adaptive attack on this defense is worse than only attacking the model.
    params = ['strength', 'radius', 'center']

    def __init__(self, randomized: bool, strength: float = None, radius: int = None, center: Tuple[int, int] = None):
        super().__init__(**get_params())

    # noinspection PyMethodOverriding
    def _forward_one(self, x: torch.Tensor, strength: float, radius: int, center: Tuple[int, int]) -> torch.Tensor:
        x_np = x.cpu().numpy()
        x_np = transform.swirl(x_np, center=center, strength=strength, radius=radius)
        return torch.from_numpy(x_np).type_as(x)

    def get_random_params(self) -> dict:
        params = {
            'strength': (2.0 - 0.01) * np.random.random(1).item() + 0.01,
            'radius': np.random.randint(low=10, high=201),
            'center': np.random.randint(low=1, high=32, size=2),
        }
        return params
