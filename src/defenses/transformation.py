import kornia
import skimage
import torch
import torchvision.transforms.functional as F

from src.defenses.base import InstancePreprocessorPyTorch, bpda_identity
from src.defenses.utils import rand
from src.utils.typing import FLOAT_INTERVAL, INT_INTERVAL


class Median(InstancePreprocessorPyTorch):
    params = ['kernel_size']

    def __init__(self, kernel_size: INT_INTERVAL = (2, 14)):
        super().__init__()
        self.kernel_size = kernel_size

    @bpda_identity
    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        kernel_size = rand(self.kernel_size, size=2) | 1  # ensure odd
        x = kornia.filters.median_blur(x[None], kernel_size=tuple(kernel_size.tolist()))
        return x[0]


class Gaussian(InstancePreprocessorPyTorch):
    params = ['kernel_size', 'sigma']

    def __init__(self, kernel_size: INT_INTERVAL = (0, 13), sigma: FLOAT_INTERVAL = (0.1, 3.1)):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    @bpda_identity
    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        kernel_size = rand(self.kernel_size, size=2) | 1  # ensure odd
        sigma = rand(self.sigma, size=2)
        x = kornia.filters.gaussian_blur2d(x[None], kernel_size=tuple(kernel_size.tolist()), sigma=tuple(sigma))
        return x[0]


class Swirl(InstancePreprocessorPyTorch):
    params = ['strength', 'radius', 'center']

    def __init__(
        self,
        strength: FLOAT_INTERVAL = (0.01, 2.00),
        radius: INT_INTERVAL = (10, 201),
        center: INT_INTERVAL = (1, 201),
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


class ResizePad(InstancePreprocessorPyTorch):
    params = ['in_size', 'out_size']

    def __init__(self, in_size: int = 224, out_size: int = 256):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == x.shape[2] == self.in_size

        # randomly resize to [in_size, out_size).
        resize_to = rand((self.in_size, self.out_size))
        x = F.resize(x, [resize_to, resize_to])

        # pad the remaining pixels
        rem = self.out_size - resize_to
        pl = rand((0, rem))
        pr = rem - pl
        pt = rand((0, rem))
        pb = rem - pt
        x = F.pad(x, [pl, pt, pr, pb])

        assert x.shape[1] == x.shape[2] == self.out_size

        return x


class Crop(InstancePreprocessorPyTorch):
    params = ['in_size', 'crop_size']

    def __init__(self, in_size: int = 224, crop_size: int = 128):
        super().__init__()
        self.in_size = in_size
        self.crop_size = crop_size

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        max_xy = self.in_size - self.crop_size - 1
        pt, pl = rand((0, max_xy), size=2)
        x = F.crop(x, top=pt, left=pl, height=self.crop_size, width=self.crop_size)
        return x
