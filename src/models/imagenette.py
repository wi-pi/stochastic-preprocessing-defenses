import torch.nn as nn
from torchvision import models

from src.models.layers import NormalizationLayer
from src.models.base import AbstractBaseResNet


IMAGENETTE_MODELS = {
    'clean': 'static/models/imagenette/clean/checkpoints/epoch53-acc0.969.ckpt',
    'var0.05': 'static/models/imagenette/var0.05/checkpoints/last.ckpt',
    'var0.10': 'static/models/imagenette/var0.10/checkpoints/last.ckpt',
    'var0.15': 'static/models/imagenette/var0.15/checkpoints/last.ckpt',
    'var0.20': 'static/models/imagenette/var0.20/checkpoints/last.ckpt',
    'var0.25': 'static/models/imagenette/var0.25/checkpoints/last.ckpt',
    'var0.30': 'static/models/imagenette/var0.30/checkpoints/last.ckpt',
}

class ImageNetteResNet(AbstractBaseResNet):

    @staticmethod
    def make_model() -> nn.Module:
        normalize = NormalizationLayer.preset('imagenet')
        model = models.resnet34(pretrained=True)
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=10)
        return nn.Sequential(normalize, model)
