"""
This script contains static files for each defense.
"""
from torchvision.datasets.utils import download_url

MODELS = [
    'https://github.com/wielandbrendel/robustness_workshop/releases/download/v0.0.1/kwta_spresnet18_0.1_cifar_adv.pth',
]


if __name__ == '__main__':
    for url in MODELS:
        download_url(url, './static/models/')
