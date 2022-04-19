from functools import partial

import torch
import torch.nn as nn
from torchvision import models

from src.models.base import AbstractBaseResNet


def load_pl_models(model: nn.Module, weight_file: str):
    state_dict = torch.load(weight_file, map_location='cpu')['state_dict']
    state_dict = {k.removeprefix('model.'): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model


def resnet18(weight_file: str | None = None):
    model = models.resnet18(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    if weight_file:
        load_pl_models(model, weight_file)
    return model


CIFAR10_MODELS = {
    'r18': partial(resnet18, 'static/models/cifar10/resnet18/checkpoints/epoch38-acc0.929.ckpt'),
    'r18.aug3': partial(resnet18, 'static/models/cifar10/augment3/checkpoints/epoch19-acc0.768.ckpt'),
    'r18.aug4': partial(resnet18, 'static/models/cifar10/augment4/checkpoints/epoch18-acc0.752.ckpt'),
}


class CIFAR10ResNet(AbstractBaseResNet):

    @staticmethod
    def make_model() -> nn.Module:
        return resnet18()
