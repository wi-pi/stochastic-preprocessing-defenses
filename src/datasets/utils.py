import numpy as np

import torch


class ToNumpy(object):

    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.numpy().astype(self.dtype)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(dtype={self.dtype})'
