import random
from functools import partial
from typing import Sequence

import numpy as np
import skimage
import torch
from scipy import fftpack

from src.defenses.base import InstancePreprocessorPyTorch, bpda_identity
from src.defenses.utils import rand
from src.utils.typing import FLOAT_INTERVAL, INT_INTERVAL


class NoiseInjection(InstancePreprocessorPyTorch):
    all_modes = ('gaussian', 'poisson', 'salt', 'pepper', 's&p', 'speckle')
    params = ['mode']

    def __init__(self, mode: Sequence[str] = all_modes):
        super().__init__()
        self.mode = mode

    @bpda_identity
    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().cpu().numpy()
        x_np = skimage.util.random_noise(x_np, mode=random.choice(self.mode))
        return torch.from_numpy(x_np).type_as(x)


class NoiseInjectionPyTorch(InstancePreprocessorPyTorch):
    all_modes = ('gaussian', 'poisson', 'salt', 'pepper', 's&p', 'speckle')
    params = ['mode']

    def __init__(self, mode: Sequence[str] = all_modes):
        super().__init__()
        self.mode = mode

    @bpda_identity
    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        mode = random.choice(self.mode)
        return self.random_noise(x, mode)

    def random_noise(self, x: torch.Tensor, mode: str):
        """PyTorch version of skimage.util.random_noise. """
        add_noise = {
            'gaussian': self.gaussian,
            'poisson': self.poisson,
            'salt': partial(self.salt_and_pepper, salt_vs_pepper=1),
            'pepper': partial(self.salt_and_pepper, salt_vs_pepper=0),
            's&p': partial(self.salt_and_pepper, salt_vs_pepper=0.5),
            'speckle': self.speckle,
        }[mode]
        return torch.clip(add_noise(x), min=0, max=1)

    @staticmethod
    def gaussian(x: torch.Tensor):
        return x + torch.randn_like(x) * 0.1  # N(0, 1) * 0.1 = N(0, 0.1)

    @staticmethod
    def poisson(x: torch.Tensor):
        vals = len(torch.unique(x))
        vals = 2 ** np.ceil(np.log2(vals))
        return torch.poisson(x * vals) / vals

    @staticmethod
    def salt_and_pepper(x: torch.Tensor, salt_vs_pepper: float):
        def _bernoulli(p: float):
            if p == 0:
                return torch.zeros_like(x).bool()
            if p == 1:
                return torch.ones_like(x).bool()
            return torch.bernoulli(x, p=p).bool()

        flipped = _bernoulli(p=0.05)
        salted = _bernoulli(p=salt_vs_pepper)
        peppered = ~salted

        out = x.clone()
        out[flipped & salted] = 1.0
        out[flipped & peppered] = 0
        return out

    @staticmethod
    def speckle(x: torch.Tensor):
        return x + x * torch.randn_like(x) * 0.1


class GaussianNoisePyTorch(InstancePreprocessorPyTorch):
    params = ['variance']

    def __init__(self, variance: float = 1.0):
        super().__init__()
        self.variance = variance

    @bpda_identity
    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        x = x + torch.randn_like(x) * self.variance
        return torch.clip(x, min=0, max=1)


class FFTPerturbation(InstancePreprocessorPyTorch):
    params = ['mask', 'fraction']

    def __init__(self, mask: INT_INTERVAL = (0, 2), fraction: FLOAT_INTERVAL = (0.00, 0.95)):
        super().__init__()
        self.mask = mask
        self.fraction = fraction

    @bpda_identity
    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        mask, fraction = rand(self.mask, size=3), rand(self.fraction, size=3)
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
