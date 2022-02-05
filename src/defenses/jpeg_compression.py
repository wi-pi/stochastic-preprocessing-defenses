import torch

from src.defenses.base import Preprocessor


def _compress(x: torch.Tensor, quality: int) -> torch.Tensor:
    import io
    from PIL import Image
    import torchvision.transforms.functional as F

    buf = io.BytesIO()
    F.to_pil_image(x).save(buf, format='jpeg', quality=quality)
    x_jpeg = F.to_tensor(Image.open(buf)).type_as(x)
    buf.close()
    return x_jpeg


class JpegCompression(Preprocessor):

    def __init__(self, quality: int = 50):
        super().__init__()
        self.quality = quality

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x_jpeg = torch.zeros_like(x)
        for i in range(x.shape[0]):
            x_jpeg[i] = _compress(x[i], self.quality)

        return x_jpeg

    def _estimate_forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
