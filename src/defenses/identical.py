"""
This module implements defenses that DO NOT require fine-tuning the model.
"""
from random import randint
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as nnf
import torchvision.transforms.functional as F
from scipy.fft import dctn, idctn

from src.defenses.base import DEFENSES, RandomizedPreprocessor
from src.utils.registry import get_params


@DEFENSES
class Empty(RandomizedPreprocessor):

    def __init__(self, randomized: bool):
        super().__init__(**get_params())

    def _forward_one(self, x: torch.Tensor, **params) -> torch.Tensor:
        return x

    def get_random_params(self) -> dict:
        return {}


@DEFENSES
class ResizePad(RandomizedPreprocessor):
    params = ['add', 'pad']

    def __init__(
        self,
        randomized: bool,
        in_size: int = 224,
        out_size: int = 256,
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
        add = randint(1, 32)
        params = {
            'add': add,
            'pad': [randint(0, add), randint(0, add)],
        }
        return params


@DEFENSES
class Crop(RandomizedPreprocessor):
    params = ['xy']

    def __init__(self, randomized: bool, in_size: int = 224, crop_size: int = 128, sx: int = 0, sy: int = 0):
        super().__init__(**get_params())
        self.in_size = in_size
        self.crop_size = crop_size

    # noinspection PyMethodOverriding
    def _forward_one(self, x: torch.Tensor, xy: Tuple[int, int]) -> torch.Tensor:
        x = F.crop(x, top=xy[0], left=xy[1], height=self.crop_size, width=self.crop_size)
        return x

    def _estimate_forward_one(self, x: torch.Tensor, **params) -> torch.Tensor:
        return self._forward_one(x, **params)

    def get_random_params(self) -> dict:
        max_xy = self.in_size - self.crop_size - 1
        params = {
            'xy': [randint(0, max_xy), randint(0, max_xy)]
        }
        return params


@DEFENSES
class DCT(RandomizedPreprocessor):
    params = []
    size = 8

    def __init__(self, randomized: bool):
        super().__init__(**get_params())
        self.q = self.get_q_table(self.size)

    # noinspection PyMethodOverriding
    def _forward_one(self, x: torch.Tensor) -> torch.Tensor:
        # record original input
        x_ori = x

        # pad to 8x
        pad = -(x.shape[1] % -self.size)
        p1 = pad // 2
        p2 = pad - p1
        x = F.pad(x, [p1, p1, p2, p2])
        c, h, w = x.shape

        # to blocks
        x = x.cpu()
        for dim in [1, 2]:
            x = x.unfold(dim, size=self.size, step=self.size)

        # DCT Quantization
        x = x.numpy()
        blocks_dct = dctn(x, axes=(-2, -1), norm='ortho')
        blocks_dct = np.round(blocks_dct * 255 / self.q) * self.q
        x = idctn(blocks_dct, axes=(-2, -1), norm='ortho') / 255
        x = torch.from_numpy(x)

        # to image
        x = x.reshape(c, -1, self.size * self.size).permute(0, 2, 1)
        x = nnf.fold(x, output_size=(h, w), kernel_size=self.size, stride=self.size)
        x = x.squeeze().type_as(x_ori)

        # crop
        x = F.crop(x, p1, p1, x_ori.shape[1], x_ori.shape[2])

        return x

    def get_random_params(self) -> dict:
        return {}

    @staticmethod
    def get_q_table(size: int = 8):
        # https://github.com/YiZeng623/Advanced-Gradient-Obfuscating/blob/master/defense.py#L165-L166
        q = np.ones((size, size), dtype=np.float32) * 30
        q[:4, :4] = 25
        return q
