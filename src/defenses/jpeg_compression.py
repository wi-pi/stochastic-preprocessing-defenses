from random import randint

import torch

from src.defenses.base import DEFENSES, RandomizedPreprocessor
from src.utils import get_params


def _compress(x: torch.Tensor, quality: int) -> torch.Tensor:
    import io
    from PIL import Image
    import torchvision.transforms.functional as F

    buf = io.BytesIO()
    F.to_pil_image(x).save(buf, format='jpeg', quality=quality)
    x_jpeg = F.to_tensor(Image.open(buf)).type_as(x)
    buf.close()
    return x_jpeg


@DEFENSES
class JpegCompression(RandomizedPreprocessor):
    params = ['quality']

    def __init__(self, randomized: bool, quality: int = 50):
        super().__init__(**get_params())

    def _forward_one(self, x: torch.Tensor, **params) -> torch.Tensor:
        return _compress(x, **params)

    def _estimate_forward_one(self, x: torch.Tensor, **params) -> torch.Tensor:
        return x

    def get_random_params(self) -> dict:
        params = {
            'quality': randint(55, 75),
        }
        return params
