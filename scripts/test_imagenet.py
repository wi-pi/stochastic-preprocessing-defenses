import argparse
import os.path
from functools import partial

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
from src.utils.gpu import setgpu
from src.utils.testkit import BaseTestKit

# https://pytorch.org/vision/stable/models.html
PRETRAINED_MODELS = {
    'r18': models.resnet18,  # acc = 69.50
    'r50': models.resnet50,  # acc = 75.92
    'inception': models.inception_v3,  # acc = 77.18
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
    parser.add_argument('--data-dir', type=str, default='static/datasets/imagenet')
    parser.add_argument('--data-skip', type=int, default=50)
    # attack
    parser.add_argument('--norm', type=str, default='inf')
    parser.add_argument('--eps', type=float, default=8)
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--step', type=int, default=10)
    parser.add_argument('-t', '--target', type=int, default=-1)
    # auto pgd
    parser.add_argument('-a', '--attack', type=str, default='pgd', choices=['pgd', 'auto', 'aggmo'])
    parser.add_argument('--random-init', type=int, default=1)
    # defense
    parser.add_argument('-d', '--defenses', type=str, choices=DEFENSES, nargs='+')
    parser.add_argument('-k', type=int)
    parser.add_argument('--eot', type=int, default=1)
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
    cache_file = f'./static/imagenet.{args.data_skip}.npz'
    if not os.path.exists(cache_file):
        transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), lambda x: x.numpy().astype(np.float32)])
        dataset = ImageFolder(args.data_dir, transform=transform)
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
    msg = 'Attack: norm {norm}, eps {eps:.5f}, eps_step {eps_step:.5f}, step {max_iter}, targeted {targeted}.'
    logger.debug(msg.format_map(pgd_kwargs))
    match args.attack:
        case 'pgd':
            logger.debug('Using PGD.')
            attack_fn = partial(PGD, **pgd_kwargs)
        case 'auto':
            logger.debug('Using Auto PGD.')
            attack_fn = partial(APGD, **pgd_kwargs, nb_random_init=args.random_init)
        case 'aggmo':
            logger.debug('Using AggMo PGD.')
            attack_fn = partial(AggMoPGD, **pgd_kwargs, b=6)
        case _:
            raise NotImplementedError(args.attack)

    # Load defense
    defense = load_defense(args.defenses)
    logger.debug(f'Defense: {defense}.')

    # Load test
    testkit_cls = TestKitForAggMo if args.attack == 'aggmo' else TestKit
    testkit = testkit_cls(model, defense, attack_fn, args.batch, args.repeat)

    if targeted:
        logger.debug(f'Test with target {args.target}.')
        y_target = np.zeros_like(y_test) + args.target
        testkit.test_targeted(x_test, y_target, test_non_adaptive=True, eot_samples=args.eot, mode=args.mode)
    else:
        testkit.test_untargeted(x_test, y_test, test_non_adaptive=True, eot_samples=args.eot, mode=args.mode)


if __name__ == '__main__':
    main(parse_args())
