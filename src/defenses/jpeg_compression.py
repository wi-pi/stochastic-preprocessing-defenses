from random import randint

import torch

from src.defenses.base import DEFENSES, RandomizedPreprocessor


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

    @classmethod
    def as_fixed(cls, quality: int = 50):
        return cls(randomized=False, quality=quality)

    def _forward_one(self, x: torch.Tensor, **params) -> torch.Tensor:
        return _compress(x, **params)

    def _estimate_forward_one(self, x: torch.Tensor, **params) -> torch.Tensor:
        return x

    @staticmethod
    def get_random_params() -> dict:
        params = {
            'quality': randint(55, 75),
        }
        return params
