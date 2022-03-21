import torch
from torchvision import models


def smoothing_model(pretrained: bool, sigma: float):
    model = models.resnet50(pretrained=False)
    weight_file = f'static/models/smoothing-models/imagenet/resnet50/noise_{sigma:.2f}/checkpoint.pth.tar'
    state_dict = torch.load(weight_file, map_location='cpu')['state_dict']
    state_dict = {k.removeprefix('1.module.'): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model
