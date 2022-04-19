import torch.nn as nn
from torchvision import models

from src.models.base import AbstractBaseResNet


class ImageNetteResNet(AbstractBaseResNet):

    @staticmethod
    def make_model() -> nn.Module:
        return models.resnet34(pretrained=False, num_classes=10)
