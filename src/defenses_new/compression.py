import io

import numpy as np
import torch
import torch.nn.functional as nnf
import torchvision.transforms.functional as F
from PIL import Image
from scipy.fft import dctn, idctn

from src.defenses_new.base import InstancePreprocessorPyTorch
from src.defenses_new.utils import bpda_identity, rand, rand_t
from src.utils.typing import INT_INTERVAL


class Quantization(InstancePreprocessorPyTorch):

    def __init__(self, scale: INT_INTERVAL = (8, 200)):
        super().__init__()
        self.scale = scale

    @bpda_identity
    def forward_one(self, x_t: torch.Tensor) -> torch.Tensor:
        scale = rand_t(self.scale, size=(3, 1, 1)).type_as(x_t)
        x_t = torch.round(x_t * scale) / scale
        return x_t


class JPEG(InstancePreprocessorPyTorch):

    def __init__(self, quality: INT_INTERVAL = (55, 75)):
        super().__init__()
        self.quality = quality

    @bpda_identity
    def forward_one(self, x_t: torch.Tensor) -> torch.Tensor:
        buf = io.BytesIO()
        F.to_pil_image(x_t).save(buf, format='jpeg', quality=rand(self.quality))
        x_t = F.to_tensor(Image.open(buf)).type_as(x_t)
        buf.close()
        return x_t


class DCT(InstancePreprocessorPyTorch):

    def __init__(self, blk_size: int = 8):
        super().__init__()
        self.size = blk_size
        self.q = self.get_q_table(blk_size)

    @bpda_identity
    def forward_one(self, x_t: torch.Tensor) -> torch.Tensor:
        # record original input
        x_ori = x_t

        # pad to 8x
        pad = -(x_t.shape[1] % -self.size)
        p1 = pad // 2
        p2 = pad - p1
        x_t = F.pad(x_t, [p1, p1, p2, p2])
        c, h, w = x_t.shape

        # to blocks
        x_t = x_t.cpu()
        for dim in [1, 2]:
            x_t = x_t.unfold(dim, size=self.size, step=self.size)

        # DCT Quantization
        x_t = x_t.numpy()
        blocks_dct = dctn(x_t, axes=(-2, -1), norm='ortho')
        blocks_dct = np.round(blocks_dct * 255 / self.q) * self.q
        x_t = idctn(blocks_dct, axes=(-2, -1), norm='ortho') / 255
        x_t = torch.from_numpy(x_t)

        # to image
        x_t = x_t.reshape(c, -1, self.size * self.size).permute(0, 2, 1)
        x_t = nnf.fold(x_t, output_size=(h, w), kernel_size=self.size, stride=self.size)
        x_t = x_t.squeeze().type_as(x_ori)

        # crop
        x_t = F.crop(x_t, p1, p1, x_ori.shape[1], x_ori.shape[2])

        return x_t

    @staticmethod
    def get_q_table(size: int = 8):
        # https://github.com/YiZeng623/Advanced-Gradient-Obfuscating/blob/master/defense.py#L165-L166
        q = np.ones((size, size), dtype=np.float32) * 30
        q[:4, :4] = 25
        return q
