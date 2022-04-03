"""
This script contains static files for each defense.
"""
import os

from torchvision.datasets.utils import download_url

MODELS = [
    'https://github.com/wielandbrendel/robustness_workshop/releases/download/v0.0.1/kwta_spresnet18_0.1_cifar_adv.pth',
]

ZIP_MODELS = {
    'natural.zip': 'https://www.dropbox.com/s/cgzd5odqoojvxzk/natural.zip?dl=1',
}

if __name__ == '__main__':
    for url in MODELS:
        download_url(url, './static/models/')

    for name, url in ZIP_MODELS.items():
        os.system(f'wget -O "./static/models/{name}" "{url}" && unzip "./static/models/{name}" -d "./static/models/"')
