import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch.nn as nn
from art.attacks.evasion import AutoProjectedGradientDescent as APGD, ProjectedGradientDescentPyTorch as PGD
from art.defences.preprocessor.preprocessor import PreprocessorPyTorch
from art.estimators.classification import PyTorchClassifier
from loguru import logger

from configs import DEFENSES, load_defense
from src.art_extensions.attacks import AggMoPGD
from src.art_extensions.classifiers import loss_gradient_average_logits
from src.datasets import ImageNet
from src.models import IMAGENET_MODELS, NormalizationLayer, LinearLoss
from src.utils.gpu import setgpu
from src.utils.testkit import BaseTestKit


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


# noinspection DuplicatedCode
def parse_args():
    parser = argparse.ArgumentParser()
    # basic
    parser.add_argument('--load', type=str, default='r18', choices=IMAGENET_MODELS)
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
    parser.add_argument('-d', '--defenses', type=str, choices=DEFENSES, nargs='+', default=[])
    parser.add_argument('-k', '--nb-defenses', type=int)
    parser.add_argument('-p', '--params', nargs='+', help='additional kwargs passed to defenses')
    args = parser.parse_args()
    return args


# noinspection DuplicatedCode
def main(args):
    # Basic
    setgpu(args.gpu, gb=10.0)
    targeted = args.target >= 0
    match args.norm:
        case 'inf':
            args.eps /= 255
            args.lr /= 255
        case _:
            args.norm = int(args.norm)

    # Load data
    cache_file = args.data_dir / f'imagenet.{args.data_skip}.npz'
    if not cache_file.exists():
        dataset = ImageNet(root_dir=args.data_dir / 'val', transform='resnet', skip=args.data_skip)
        x_test, y_test = map(np.stack, zip(*dataset))
        np.savez(cache_file, x_test, y_test)

    x_test, y_test = np.load(cache_file).values()
    logger.debug(f'Loaded dataset x: {x_test.shape}, y: {y_test.shape}.')

    # Load model
    model = nn.Sequential(
        NormalizationLayer.preset('imagenet'),
        IMAGENET_MODELS[args.load](pretrained=True),
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
    match args.attack:
        case 'aggmo':
            testkit_cls = TestKitForAggMo
        case _:
            testkit_cls = TestKit

    testkit = testkit_cls(model, defense, attack_fn, batch_size=args.batch, mode=args.mode, nb_repeats=args.repeat)

    # Run test
    y_reference = np.full_like(y_test, args.target) if targeted else y_test
    testkit.test(x_test, y_reference, targeted=targeted, test_non_adaptive=args.test_non_adaptive, eot=args.eot)


if __name__ == '__main__':
    main(parse_args())
