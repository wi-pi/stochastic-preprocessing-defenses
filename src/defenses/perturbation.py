import random
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
