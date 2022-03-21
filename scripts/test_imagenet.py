import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from art.attacks.evasion import AutoProjectedGradientDescent as APGD, ProjectedGradientDescentPyTorch as PGD
from art.defences.preprocessor.preprocessor import PreprocessorPyTorch
from art.estimators.classification import PyTorchClassifier
from loguru import logger
from torchvision import models
from torchvision.datasets import ImageFolder
from tqdm import trange

from configs import DEFENSES, load_defense
from src.art_extensions.attacks import AggMoPGD
from src.art_extensions.classifiers import loss_gradient_average_logits
from src.models.layers import NormalizationLayer
from src.models.loss import LinearLoss
from src.models.smooth_models import smoothing_model
from src.utils.gpu import setgpu
from src.utils.testkit import BaseTestKit

# suggested batch size:
# r18: 250
# r50: 100
PRETRAINED_MODELS = {
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


class TestKit(BaseTestKit):

    @staticmethod
    def get_estimator(model: nn.Module, defense: PreprocessorPyTorch | None = None) -> PyTorchClassifier:
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


class TestKitForAggMo(BaseTestKit):
    @staticmethod
    def get_estimator(model: nn.Module, defense: PreprocessorPyTorch | None = None) -> PyTorchClassifier:
        wrapper = PyTorchClassifier(
            model,
            loss=LinearLoss(),
            input_shape=(3, 224, 224),
            nb_classes=1000,
            clip_values=(0, 1),
            preprocessing=None,
            preprocessing_defences=defense,
        )
        return wrapper

    @staticmethod
    def prepare_estimator_for_eot(estimator: PyTorchClassifier, eot_samples: int = 1):
        estimator.loss_gradient = loss_gradient_average_logits(estimator, nb_samples=eot_samples)


def parse_args():
    parser = argparse.ArgumentParser()
    # basic
    parser.add_argument('--load', type=str, default='r18', choices=PRETRAINED_MODELS)
    parser.add_argument('-b', '--batch', type=int, default=250)
    parser.add_argument('-g', '--gpu', type=int)
    parser.add_argument('-m', '--mode', type=str, default='all', choices=['all', 'any', 'vote'])
    parser.add_argument('-n', '--repeat', type=int, default=10)
    # dataset
    parser.add_argument('--data-dir', type=Path, default='static/datasets/imagenet')
    parser.add_argument('--data-skip', type=int, default=50)
    # attack
    parser.add_argument('--norm', type=str, default='inf')
    parser.add_argument('--eps', type=float, default=8)
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--step', type=int, default=10)
    parser.add_argument('--eot', type=int, default=1)
    parser.add_argument('-t', '--target', type=int, default=-1)
    parser.add_argument('--test-non-adaptive', action='store_true')
    # auto pgd
    parser.add_argument('-a', '--attack', type=str, default='pgd', choices=['pgd', 'auto', 'aggmo'])
    parser.add_argument('--random-init', type=int, default=1)
    # defense
    parser.add_argument('-d', '--defenses', type=str, choices=DEFENSES, nargs='+')
    parser.add_argument('-k', '--nb-defenses', type=int)
    parser.add_argument('-p', '--params', nargs='+', help='additional kwargs passed to defenses')
    args = parser.parse_args()
    return args


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
    cache_file = args.data_dir / f'imagenet.{args.data_skip}.npz'
    if not cache_file.exists():
        transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), lambda x: x.numpy().astype(np.float32)])
        dataset = ImageFolder(args.data_dir / 'val', transform=transform)
        subset = [dataset[i] for i in trange(0, len(dataset), args.data_skip, desc='Load dataset', leave=False)]
        x_test, y_test = map(np.stack, zip(*subset))
        np.savez(cache_file, x_test=x_test, y_test=y_test)
    else:
        subset = np.load(cache_file)
        x_test, y_test = subset['x_test'], subset['y_test']

    logger.debug(f'Loaded dataset x: {x_test.shape}, y: {y_test.shape}.')

    # Load model
    model = nn.Sequential(
        NormalizationLayer.preset('imagenet'),
        PRETRAINED_MODELS[args.load](pretrained=True),
    )
    logger.debug(f'Loaded model from "{args.load}".')

    # Load attack
    pgd_kwargs = dict(norm=args.norm, eps=args.eps, eps_step=args.lr, max_iter=args.step, targeted=targeted)
    msg = 'Attack ({name}): norm {norm}, eps {eps:.5f}, eps_step {eps_step:.5f}, step {max_iter}, targeted {targeted}.'
    logger.debug(msg.format_map(pgd_kwargs | {'name': args.attack}))
    match args.attack:
        case 'pgd':
            attack_fn = partial(PGD, **pgd_kwargs)
        case 'auto':
            attack_fn = partial(APGD, **pgd_kwargs, nb_random_init=args.random_init)
        case 'aggmo':
            attack_fn = partial(AggMoPGD, **pgd_kwargs, b=6)
        case _:
            raise NotImplementedError(args.attack)

    # Load defense
    defense = load_defense(defenses=args.defenses, nb_samples=args.nb_defenses, params=args.params)
    logger.debug(f'Defense: {defense}.')

    # Load test
    testkit_cls = TestKitForAggMo if args.attack == 'aggmo' else TestKit
    testkit = testkit_cls(model, defense, attack_fn, args.batch, args.mode, args.repeat)

    # Run test
    y_reference = np.full_like(y_test, args.target) if targeted else y_test
    testkit.test(x_test, y_reference, targeted=targeted, test_non_adaptive=args.test_non_adaptive, eot=args.eot)


if __name__ == '__main__':
    main(parse_args())
