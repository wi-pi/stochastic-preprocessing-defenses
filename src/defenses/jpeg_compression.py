import io
from random import randint

import torch
import torchvision.transforms.functional as F
from PIL import Image

from src.defenses.base import DEFENSES, RandomizedPreprocessor
from src.utils import get_params


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
