from typing import List

import numpy as np
import torch

from src.defenses.base import DEFENSES, RandomizedPreprocessor
from src.utils import get_params


@DEFENSES
class ColorReduction(RandomizedPreprocessor):
    params = ['scales']

    def __init__(self, randomized: bool, scales: List[int] = None):
        super().__init__(**get_params())

    # noinspection PyMethodOverriding
    def _forward_one(self, x: torch.Tensor, scales: List[int]) -> torch.Tensor:
        scales = torch.as_tensor(scales).type_as(x)
        x = torch.round(x * scales) / scales
        return x

    # noinspection PyMethodOverriding
    def _estimate_forward_one(self, x: torch.Tensor, scales: List[int]) -> torch.Tensor:
        return x

    def get_random_params(self) -> dict:
        if np.random.randint(2):
            scales = np.random.randint(low=8, high=200, size=(3, 1, 1))
        else:
            scales = np.random.randint(low=8, high=200, size=(1, 1, 1)).repeat(3, 0)

        params = {'scales': scales}
        return params
