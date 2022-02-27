import argparse
import os.path
from functools import partial
from typing import Optional

import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from art.attacks.evasion import ProjectedGradientDescentPyTorch as PGD
from art.defences.preprocessor.preprocessor import PreprocessorPyTorch
from art.estimators.classification import PyTorchClassifier
from loguru import logger
from torchvision import models
from torchvision.datasets import ImageFolder
from tqdm import trange

from src.defenses import *
from src.models.layers import NormalizationLayer
from src.utils.gpu import setgpu
from src.utils.testkit import BaseTestKit

# https://pytorch.org/vision/stable/models.html
PRETRAINED_MODELS = {
    'r18': models.resnet18,  # acc = 69.50
    'r50': models.resnet50,  # acc = 75.92
    'inception': models.inception_v3,  # acc = 77.18
}

DEFENSES = {cls.__name__: cls for cls in InstancePreprocessorPyTorch.__subclasses__()}


class TestKit(BaseTestKit):

    @staticmethod
    def get_estimator(model: nn.Module, defense: Optional[PreprocessorPyTorch] = None) -> PyTorchClassifier:
        wrapper = PyTorchClassifier(
            model,
            loss=nn.CrossEntropyLoss(),
            input_shape=(3, 224, 224),
            nb_classes=1000,
            clip_values=(0, 1),
            preprocessing=None,
            preprocessing_defences=defense,
        )
        return wrapper


def parse_args():
    parser = argparse.ArgumentParser()
    # basic
    parser.add_argument('--load', type=str, default='r18', choices=PRETRAINED_MODELS)
    parser.add_argument('-b', '--batch', type=int, default=250)
    parser.add_argument('-g', '--gpu', type=int)
    parser.add_argument('-n', type=int, default=10)
    # dataset
    parser.add_argument('--data-dir', type=str, default='static/datasets/imagenet')
    parser.add_argument('--data-skip', type=int, default=50)
    # attack
    parser.add_argument('--norm', type=str, default='inf')
    parser.add_argument('--eps', type=float, default=8)
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--step', type=int, default=10)
    parser.add_argument('-t', '--target', type=int, default=-1)
    # defense
    parser.add_argument('-d', '--defenses', type=str, choices=DEFENSES, nargs='+')
    parser.add_argument('-k', type=int)
    parser.add_argument('--eot', type=int, default=1)
    args = parser.parse_args()
    return args


def load_defense(args):
    """Single defense"""
    # defense = ResizePad(in_size=224, out_size=256)
    # defense = Crop(in_size=224, crop_size=128)
    # defense = DCT()
    # defense = Gaussian(kernel_size=(0, 6), sigma=(0.1, 1.1))

    """Randomized ensemble of all"""
    # defense = Ensemble(preprocessors=[DEFENSES[p]() for p in args.defenses], k=args.k)

    """Manually specified ensemble"""
    defense = Ensemble(
        preprocessors=[
            Gaussian(kernel_size=(0, 6), sigma=(1.0, 2.0)),
            Median(kernel_size=(0, 6)),
            JPEG(quality=(55, 75)),
        ],
        k=3,
    )
    return defense


# noinspection DuplicatedCode
def main(args):
    # Basic
    setgpu(args.gpu, gb=10.0)
    targeted = args.target >= 0
    if args.norm == 'inf':
        args.eps /= 255
        args.lr /= 255
    else:
        args.norm = int(args.norm)

    # Load data
    cache_file = f'./static/imagenet.{args.data_skip}.npz'
    if not os.path.exists(cache_file):
        transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), lambda x: x.numpy().astype(np.float32)])
        dataset = ImageFolder(args.data_dir, transform=transform)
        subset = [dataset[i] for i in trange(0, len(dataset), args.data_skip, desc='Load dataset', leave=False)]
        x_test = np.stack([x for x, y in subset])
        y_test = np.array([y for x, y in subset])
        np.savez(cache_file, x_test=x_test, y_test=y_test)
    else:
        subset = np.load(cache_file)
        x_test = subset['x_test']
        y_test = subset['y_test']

    logger.debug(f'Loaded dataset x: {x_test.shape}, y: {y_test.shape}.')

    # Load model
    model = nn.Sequential(
        NormalizationLayer.preset('imagenet'),
        PRETRAINED_MODELS[args.load](pretrained=True),
    )
    logger.debug(f'Loaded model from "{args.load}".')

    # Load attack
    logger.debug(
        f'Attack: norm {args.norm}, eps {args.eps:.5f}, eps_step {args.lr:.5f}, step {args.step} targeted {targeted}.')
    attack_fn = partial(PGD, norm=args.norm, eps=args.eps, eps_step=args.lr, max_iter=args.step, targeted=targeted)

    # Load defense
    defense = load_defense(args)
    logger.debug(f'Defense: {defense}.')

    # Load test
    testkit = TestKit(model, defense, attack_fn, args.batch, args.n)

    if targeted:
        logger.debug(f'Test with target {args.target}.')
        y_target = np.zeros_like(y_test) + args.target
        testkit.test_targeted(x_test, y_target, test_non_adaptive=True, eot_samples=args.eot, mode='all')
    else:
        testkit.test_untargeted(x_test, y_test, test_non_adaptive=True, eot_samples=args.eot, mode='all')


if __name__ == '__main__':
    main(parse_args())
