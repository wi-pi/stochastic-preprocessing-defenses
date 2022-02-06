import io
from random import randint
from typing import List
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from skimage import transform

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

    def get_random_params(self) -> dict:
        if np.random.randint(2):
            scales = np.random.randint(low=8, high=200, size=(3, 1, 1))
        else:
            scales = np.random.randint(low=8, high=200, size=(1, 1, 1)).repeat(3, 0)

        params = {'scales': scales}
        return params


@DEFENSES
class JpegCompression(RandomizedPreprocessor):
    params = ['quality']

    def __init__(self, randomized: bool, quality: int = 50):
        super().__init__(**get_params())

    # noinspection PyMethodOverriding
    def _forward_one(self, x: torch.Tensor, quality: int) -> torch.Tensor:
        buf = io.BytesIO()
        F.to_pil_image(x).save(buf, format='jpeg', quality=quality)
        x_jpeg = F.to_tensor(Image.open(buf)).type_as(x)
        buf.close()
        return x_jpeg

    def get_random_params(self) -> dict:
        params = {
            'quality': randint(55, 75),
        }
        return params


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
