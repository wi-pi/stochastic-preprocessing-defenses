from random import randint
from typing import Tuple

import torch
import torchvision.transforms.functional as F

from src.defenses.base import DEFENSES, RandomizedPreprocessor
from src.utils import get_params


@DEFENSES
class ResizePad(RandomizedPreprocessor):
    params = []

    def __init__(
        self,
        randomized: bool,
        in_size: int = 32,
        out_size: int = 35,
        add: int = 3,
        pad: Tuple[int, int] = (1, 1)
    ):
        super().__init__(**get_params())
        self.in_size = in_size
        self.out_size = out_size

    # noinspection PyMethodOverriding
    def _forward_one(self, x: torch.Tensor, add: int, pad: Tuple[int, int]) -> torch.Tensor:
        assert x.shape[1] == x.shape[2] == self.in_size

        resize_to = self.in_size + add
        x = F.resize(x, [resize_to, resize_to])

        pl, pt = pad
        pr = self.out_size - resize_to - pl
        pb = self.out_size - resize_to - pt
        x = F.pad(x, [pl, pt, pr, pb])

        assert x.shape[1] == x.shape[2] == self.out_size

        return x

    def _estimate_forward_one(self, x: torch.Tensor, **params) -> torch.Tensor:
        return self._forward_one(x, **params)

    def get_random_params(self) -> dict:
        add = randint(1, 3)
        params = {
            'add': add,
            'pad': [randint(0, add), randint(0, add)],
        }
        return params
