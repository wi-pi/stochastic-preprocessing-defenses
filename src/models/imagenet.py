from functools import partial

import torch
from torchvision import models


def smoothing_model(pretrained: bool, sigma: float):
    """
    Load a ResNet-50 trained with noise.

    Download models from: https://github.com/locuslab/smoothing

    :param pretrained: Ignored.
    :param sigma: Currently only support: 0.00, 0.25, 0.50, 1.00
    :return: Loaded model.
    """
    model = models.resnet50(pretrained=False)
    weight_file = f'static/models/smoothing-models/imagenet/resnet50/noise_{sigma:.2f}/checkpoint.pth.tar'
    state_dict = torch.load(weight_file, map_location='cpu')['state_dict']
    state_dict = {k.removeprefix('1.module.'): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model


# suggested batch size:
# r18: 250
# r50: 100
IMAGENET_MODELS = {
    # https://pytorch.org/vision/stable/models.html
    'r18': models.resnet18,  # acc = 69.50
    'r50': models.resnet50,  # acc = 75.92
    'inception': models.inception_v3,  # acc = 77.18

    # https://github.com/locuslab/smoothing/tree/master
    # Smoothing models are trained with additive Gaussian noise *without* clip to (0, 1)
    'r50-s0.00': partial(smoothing_model, sigma=0.00),  # acc = 75.90 (N=20, sigma=0.00, no clip)  75.94 (clip)
    'r50-s0.25': partial(smoothing_model, sigma=0.25),  # acc = 70.00 (N=20, sigma=0.25, no clip)  70.00 (clip)
    'r50-s0.50': partial(smoothing_model, sigma=0.50),  # acc = 63.21 (N=20, sigma=0.50, no clip)
    'r50-s1.00': partial(smoothing_model, sigma=1.00),  # acc = 50.80 (N=20, sigma=1.00, no clip)
}
