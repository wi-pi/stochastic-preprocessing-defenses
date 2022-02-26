import kornia
import skimage
import torch

from src.defenses_new.base import InstancePreprocessorPyTorch, bpda_identity
from src.defenses_new.utils import rand
from src.utils.typing import FLOAT_INTERVAL, INT_INTERVAL


class Median(InstancePreprocessorPyTorch):

    def __init__(self, kernel_size: INT_INTERVAL = (0, 13)):
        super().__init__()
        self.kernel_size = kernel_size

    @bpda_identity
    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        kernel_size = rand(self.kernel_size, size=2) | 1  # ensure odd
        x = kornia.filters.median_blur(x[None], kernel_size=kernel_size)
        return x[0]


class Gaussian(InstancePreprocessorPyTorch):

    def __init__(self, kernel_size: INT_INTERVAL = (0, 13), sigma: FLOAT_INTERVAL = (0.1, 3.1)):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    @bpda_identity
    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        kernel_size = rand(self.kernel_size, size=2) | 1  # ensure odd
        sigma = rand(self.sigma, size=2)
        x = kornia.filters.gaussian_blur2d(x[None], kernel_size=kernel_size, sigma=tuple(sigma))
        return x[0]


class Swirl(InstancePreprocessorPyTorch):

    def __init__(
        self,
        strength: FLOAT_INTERVAL = (0.01, 2.00),
        radius: INT_INTERVAL = (10, 201),
        center: INT_INTERVAL = (1, 32),
    ):
        super().__init__()
        self.strength = strength
        self.radius = radius
        self.center = center

    @bpda_identity
    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        center, strength, radius = rand(self.center, size=2), rand(self.strength), rand(self.radius)
        x_np = x.detach().cpu().numpy().transpose(1, 2, 0)
        x_np = skimage.transform.swirl(x_np, center=center, strength=strength, radius=radius)
        x = torch.from_numpy(x_np.transpose(2, 0, 1)).type_as(x)
        return x
