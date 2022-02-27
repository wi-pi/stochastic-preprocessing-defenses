import io

import numpy as np
import torch
import torch.nn.functional as nnf
import torchvision.transforms.functional as F
from PIL import Image
from scipy.fft import dctn, idctn

from src.defenses.base import InstancePreprocessorPyTorch, bpda_identity
from src.defenses.utils import rand, rand_t
from src.utils.typing import INT_INTERVAL


class Quantization(InstancePreprocessorPyTorch):

    def __init__(self, scale: INT_INTERVAL = (8, 200)):
        super().__init__()
        self.scale = scale

    @bpda_identity
    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        scale = rand_t(self.scale, size=(3, 1, 1)).type_as(x)
        x = torch.round(x * scale) / scale
        return x


class JPEG(InstancePreprocessorPyTorch):

    def __init__(self, quality: INT_INTERVAL = (55, 75)):
        super().__init__()
        self.quality = quality

    @bpda_identity
    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        buf = io.BytesIO()
        F.to_pil_image(x).save(buf, format='jpeg', quality=rand(self.quality))
        x = F.to_tensor(Image.open(buf)).type_as(x)
        buf.close()
        return x


class DCT(InstancePreprocessorPyTorch):

    def __init__(self, blk_size: int = 8):
        super().__init__()
        self.size = blk_size
        self.q = self.get_q_table(blk_size)

    @bpda_identity
    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
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

    @staticmethod
    def get_q_table(size: int = 8):
        # https://github.com/YiZeng623/Advanced-Gradient-Obfuscating/blob/master/defense.py#L165-L166
        q = np.ones((size, size), dtype=np.float32) * 30
        q[:4, :4] = 25
        return q
