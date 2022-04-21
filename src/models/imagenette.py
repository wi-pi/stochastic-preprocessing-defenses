import torch.nn as nn
from torchvision import models

from src.models.layers import NormalizationLayer
from src.models.base import AbstractBaseResNet


IMAGENETTE_MODELS = {
    'clean': 'static/models/imagenette/clean-pretrain-70e/checkpoints/epoch53-acc0.969.ckpt',
}

class ImageNetteResNet(AbstractBaseResNet):

    @staticmethod
    def make_model() -> nn.Module:
        normalize = NormalizationLayer.preset('imagenet')
        model = models.resnet34(pretrained=True)
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=10)
        return nn.Sequential(normalize, model)
