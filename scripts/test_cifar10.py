import argparse
from functools import partial
from typing import Optional

import numpy as np
import torch.nn as nn
from art.attacks.evasion import ProjectedGradientDescentPyTorch as PGD
from art.defences.preprocessor.preprocessor import PreprocessorPyTorch
from art.estimators.classification import PyTorchClassifier
from loguru import logger
from torchvision.datasets import CIFAR10

from configs import DEFENSES, load_defense
from src.models.cifar10 import CIFAR10ResNet
from src.utils.gpu import setgpu
from src.utils.testkit import BaseTestKit

PRETRAINED_MODELS = {
    'common': 'static/logs/common/checkpoints/epoch38-acc0.929.ckpt',
    'augment3': 'static/logs/augment3/checkpoints/epoch19-acc0.768.ckpt',
    'augment4': 'static/logs/augment4/checkpoints/epoch18-acc0.752.ckpt',
}


class TestKit(BaseTestKit):

    @staticmethod
    def get_estimator(model: nn.Module, defense: Optional[PreprocessorPyTorch] = None) -> PyTorchClassifier:
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


def parse_args():
    parser = argparse.ArgumentParser()
    # basic
    parser.add_argument('--load', type=str, choices=PRETRAINED_MODELS, default='common')
    parser.add_argument('-b', '--batch', type=int, default=1000)
    parser.add_argument('-g', '--gpu', type=int)
    parser.add_argument('-n', type=int, default=10)
    # dataset
    parser.add_argument('--data-dir', type=str, default='static/datasets')
    parser.add_argument('--data-skip', type=int, default=10)
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


# noinspection DuplicatedCode
def main(args):
    # Basic
    setgpu(args.gpu, gb=10.0)
    targeted = args.target >= 0
    args.load = PRETRAINED_MODELS[args.load]
    if args.norm == 'inf':
        args.eps /= 255
        args.lr /= 255
    else:
        args.norm = int(args.norm)

    # Load data
    dataset = CIFAR10(args.data_dir, train=False)
    x_test = np.array(dataset.data / 255, dtype=np.float32).transpose((0, 3, 1, 2))  # to channel first
    y_test = np.array(dataset.targets, dtype=np.int)
    x_test = x_test[::args.data_skip]
    y_test = y_test[::args.data_skip]
    logger.debug(f'Loaded dataset x: {x_test.shape}, y: {y_test.shape}.')

    # Load model
    model = CIFAR10ResNet.load_from_checkpoint(args.load)
    logger.debug(f'Loaded model from "{args.load}".')

    # Load attack
    logger.debug(
        f'Attack: norm {args.norm}, eps {args.eps:.5f}, eps_step {args.lr:.5f}, step {args.step}, target {targeted}')
    attack_fn = partial(PGD, norm=args.norm, eps=args.eps, eps_step=args.lr, max_iter=args.step, targeted=targeted)

    # Load defense
    defense = load_defense(args.defenses, args.k)
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
