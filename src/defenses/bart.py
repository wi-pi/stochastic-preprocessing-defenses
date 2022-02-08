import io
import random
from random import randint
from typing import List, Tuple

import kornia
import numpy as np
import skimage.util
import torch
import torchvision.transforms.functional as F
from PIL import Image
from scipy import fftpack
from skimage import filters, morphology, transform

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
    params = ['strength', 'radius', 'center']

    def __init__(self, randomized: bool, strength: float = None, radius: int = None, center: Tuple[int, int] = None):
        super().__init__(**get_params())

    # noinspection PyMethodOverriding
    def _forward_one(self, x: torch.Tensor, strength: float, radius: int, center: Tuple[int, int]) -> torch.Tensor:
        x_np = x.cpu().numpy().transpose(1, 2, 0)
        x_np = transform.swirl(x_np, center=center, strength=strength, radius=radius)
        return torch.from_numpy(x_np.transpose(2, 0, 1)).type_as(x)

    def get_random_params(self) -> dict:
        params = {
            'strength': (2.0 - 0.01) * np.random.random(1).item() + 0.01,
            'radius': np.random.randint(low=10, high=201),
            'center': np.random.randint(low=1, high=32, size=2),
        }
        return params


@DEFENSES
class NoiseInjection(RandomizedPreprocessor):
    params = ['mode']

    def __init__(self, randomized: bool, mode: str = None):
        super().__init__(**get_params())

    # noinspection PyMethodOverriding
    def _forward_one(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        x_np = x.cpu().numpy()
        x_np = skimage.util.random_noise(x_np, mode=mode)
        return torch.from_numpy(x_np).type_as(x)

    def get_random_params(self) -> dict:
        params = {
            'mode': random.choice(['gaussian', 'poisson', 'salt', 'pepper', 's&p', 'speckle'])
        }
        return params


@DEFENSES
class FFTPerturbation(RandomizedPreprocessor):
    params = ['mask', 'fraction']

    def __init__(self, randomized: bool, mask: np.ndarray = None, fraction: np.ndarray = None):
        super().__init__(**get_params())

    # noinspection PyMethodOverriding
    def _forward_one(self, x: torch.Tensor, mask: np.ndarray, fraction: np.ndarray) -> torch.Tensor:
        x_np = x.cpu().clone().numpy().transpose(1, 2, 0)

        r, c, _ = x_np.shape
        point_factor = (1.02 - 0.98) * np.random.random((r, c)) + 0.98
        for i in range(3):
            x_np_fft = fftpack.fft2(x_np[..., i])
            r, c = x_np_fft.shape
            x_np_fft[int(r * fraction[i]):int(r * (1 - fraction[i]))] = 0
            x_np_fft[:, int(c * fraction[i]):int(c * (1 - fraction[i]))] = 0
            if mask[i]:
                m = np.ones(x_np_fft.shape[:2]) > 0
                m = ~m
                m = m * ~(np.random.uniform(size=x_np_fft.shape[:2]) < fraction[i])
                m = ~m
                x_np_fft = np.multiply(x_np_fft, m)

            x_np_fft = np.multiply(x_np_fft, point_factor)
            x_np_new = fftpack.ifft2(x_np_fft).real
            x_np_new = np.clip(x_np_new, 0, 1)
            x_np[..., i] = x_np_new

        return torch.from_numpy(x_np.transpose(2, 0, 1)).type_as(x)

    def get_random_params(self) -> dict:
        params = {
            'mask': np.random.randint(low=0, high=2, size=3),
            'fraction': 0.95 * np.random.random(3)
        }
        return params


@DEFENSES
class GaussianBlur(RandomizedPreprocessor):
    params = ['sigma']

    def __init__(self, randomized: bool, sigma: np.ndarray = None):
        super().__init__(**get_params())

    # noinspection PyMethodOverriding
    def _forward_one(self, x: torch.Tensor, sigma: np.ndarray) -> torch.Tensor:
        x_np = x.cpu().clone().numpy().transpose(1, 2, 0)
        x_np = skimage.filters.gaussian(x_np, sigma=sigma, channel_axis=2)
        return torch.from_numpy(x_np.transpose(2, 0, 1)).type_as(x)

    def get_random_params(self) -> dict:
        if np.random.randint(2):
            sigma = 0.1 + np.random.random(3) * 2
        else:
            sigma = 0.1 + np.random.random(1).repeat(3) * 2

        params = {'sigma': sigma}
        return params


@DEFENSES
class MedianBlur(RandomizedPreprocessor):
    params = ['radius']

    def __init__(self, randomized: bool, radius: np.ndarray = (2, 2, 2)):
        super().__init__(**get_params())

    # noinspection PyMethodOverriding
    def _forward_one(self, x: torch.Tensor, radius: np.ndarray) -> torch.Tensor:
        x_np = x.cpu().clone().numpy().transpose(1, 2, 0)
        for i in range(3):
            mask = morphology.disk(radius[i])
            x_np[..., i] = filters.rank.median(skimage.util.img_as_ubyte(x_np[..., i]), mask) / 255.0
        return torch.from_numpy(x_np.transpose(2, 0, 1)).type_as(x)

    def get_random_params(self) -> dict:
        if np.random.randint(2):
            radius = np.random.randint(low=2, high=5, size=3)
        else:
            radius = np.random.randint(low=2, high=5, size=1).repeat(3)

        params = {'radius': radius}
        return params


@DEFENSES
class MeanFilter(RandomizedPreprocessor):
    params = ['radius']

    def __init__(self, randomized: bool, radius: np.ndarray = (2, 2, 2)):
        super().__init__(**get_params())

    # noinspection PyMethodOverriding
    def _forward_one(self, x: torch.Tensor, radius: np.ndarray) -> torch.Tensor:
        x_np = x.cpu().clone().numpy().transpose(1, 2, 0)
        for i in range(3):
            mask = morphology.disk(radius[i])
            x_np[..., i] = filters.rank.mean(skimage.util.img_as_ubyte(x_np[..., i]), mask) / 255.0
        return torch.from_numpy(x_np.transpose(2, 0, 1)).type_as(x)

    def get_random_params(self) -> dict:
        if np.random.randint(2):
            radius = np.random.randint(low=2, high=3, size=3)
        else:
            radius = np.random.randint(low=2, high=3, size=1).repeat(3)

        params = {'radius': radius}
        return params


@DEFENSES
class Median(RandomizedPreprocessor):
    params = ['kernel_size']

    def __init__(self, randomized: bool, kernel_size: Tuple[int, int] = (3, 3)):
        super().__init__(**get_params())

    # noinspection PyMethodOverriding
    def _forward_one(self, x: torch.Tensor, kernel_size: Tuple[int, int]) -> torch.Tensor:
        x = x[None]
        x = kornia.filters.median_blur(x, kernel_size=kernel_size)
        x = x[0]
        return x

    def get_random_params(self) -> dict:
        params = {
            'kernel_size': [random.choice([3, 5]) for _ in range(2)]
        }
        return params


@DEFENSES
class Gaussian(RandomizedPreprocessor):
    params = ['kernel_size']

    def __init__(self, randomized: bool, kernel_size: Tuple[int, int] = (3, 3), sigma: Tuple[float, float] = (1, 1)):
        super().__init__(**get_params())

    # noinspection PyMethodOverriding
    def _forward_one(self, x: torch.Tensor, kernel_size: Tuple[int, int], sigma: Tuple[float, float]) -> torch.Tensor:
        x = x[None]
        x = kornia.filters.gaussian_blur2d(x, kernel_size=kernel_size, sigma=sigma)
        x = x[0]
        return x

    def get_random_params(self) -> dict:
        params = {
            'kernel_size': [random.choice([3, 5]) for _ in range(2)],
            'sigma': np.random.random(2) + 1
        }
        return params
