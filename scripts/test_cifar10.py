import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch.nn as nn
from art.attacks.evasion import ProjectedGradientDescentPyTorch as PGD
from art.defences.preprocessor.preprocessor import PreprocessorPyTorch
from art.estimators.classification import PyTorchClassifier
from loguru import logger
from torchvision.datasets import CIFAR10

from configs import DEFENSES, load_defense
from src.models import CIFAR10_MODELS
from src.utils.gpu import setgpu
from src.utils.testkit import BaseTestKit


class TestKit(BaseTestKit):

    @staticmethod
    def get_estimator(model: nn.Module, defense: PreprocessorPyTorch | None = None) -> PyTorchClassifier:
        wrapper = PyTorchClassifier(
            model,
            loss=nn.CrossEntropyLoss(),
            input_shape=(3, 32, 32),
            nb_classes=10,
            clip_values=(0, 1),
            preprocessing=None,
            preprocessing_defences=defense,
        )
        return wrapper


# noinspection DuplicatedCode
def parse_args():
    parser = argparse.ArgumentParser()
    # basic
    parser.add_argument('--load', type=str, choices=CIFAR10_MODELS, default='r18')
    parser.add_argument('-b', '--batch', type=int, default=1000)
    parser.add_argument('-g', '--gpu', type=int)
    parser.add_argument('-m', '--mode', type=str, default='all', choices=['all', 'any', 'vote'])
    parser.add_argument('-n', '--repeat', type=int, default=10)
    # dataset
    parser.add_argument('--data-dir', type=Path, default='static/datasets')
    parser.add_argument('--data-skip', type=int, default=10)
    # attack
    parser.add_argument('--norm', type=str, default='inf')
    parser.add_argument('--eps', type=float, default=8)
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--step', type=int, default=10)
    parser.add_argument('--eot', type=int, default=1)
    parser.add_argument('-t', '--target', type=int, default=-1)
    parser.add_argument('--test-non-adaptive', action='store_true')
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
    dataset = CIFAR10(args.data_dir, train=False)
    x_test = np.array(dataset.data / 255, dtype=np.float32).transpose((0, 3, 1, 2))  # to channel first
    y_test = np.array(dataset.targets, dtype=int)
    x_test = x_test[::args.data_skip]
    y_test = y_test[::args.data_skip]
    logger.debug(f'Loaded dataset x: {x_test.shape}, y: {y_test.shape}.')

    # Load model
    model = CIFAR10_MODELS[args.load]()
    logger.debug(f'Loaded model from "{args.load}".')

    # Load attack
    pgd_kwargs = dict(norm=args.norm, eps=args.eps, eps_step=args.lr, max_iter=args.step, targeted=targeted)
    msg = 'Attack: norm {norm}, eps {eps:.5f}, eps_step {eps_step:.5f}, step {max_iter}, targeted {targeted}.'
    logger.debug(msg.format_map(pgd_kwargs))
    attack_fn = partial(PGD, **pgd_kwargs)

    # Load defense
    defense = load_defense(defenses=args.defenses, nb_samples=args.nb_defenses, params=args.params)
    logger.debug(f'Defense: {defense}.')

    # Load test
    testkit = TestKit(model, defense, attack_fn, args.batch, args.mode, args.repeat)

    # Run test
    y_reference = np.full_like(y_test, args.target) if targeted else y_test
    testkit.test(x_test, y_reference, targeted=targeted, test_non_adaptive=args.test_non_adaptive, eot=args.eot)


if __name__ == '__main__':
    main(parse_args())
