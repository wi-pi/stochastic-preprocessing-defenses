import argparse
import os
from functools import partial
from typing import Optional

import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from art.attacks.evasion import ProjectedGradientDescentPyTorch as PGD
from art.defences.preprocessor.preprocessor import PreprocessorPyTorch
from loguru import logger
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from tqdm import trange

from src.art.classifier import PyTorchClassifier
from src.defenses import DEFENSES, Ensemble
from src.models.layers import NormalizationLayer
from src.utils.testkit import BaseTestKit


class TestKit(BaseTestKit):

    @staticmethod
    def get_wrapper(model: nn.Module, defense: Optional[PreprocessorPyTorch] = None):
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
    parser.add_argument('--load', type=str)
    parser.add_argument('-b', '--batch', type=int, default=250)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-n', type=int, default=10)
    # dataset
    parser.add_argument('--data-dir', type=str, default='static/datasets/imagenet')
    parser.add_argument('--data-skip', type=int, default=50)
    # attack
    parser.add_argument('--norm', type=str, default='inf')
    parser.add_argument('--eps', type=float, default=8)
    parser.add_argument('--lr', type=float, default=2)
    parser.add_argument('--step', type=int, default=10)
    # defense
    parser.add_argument('-d', '--defenses', type=str, choices=DEFENSES, nargs='+')
    parser.add_argument('-k', type=int, default=2)
    parser.add_argument('--eot', type=int, default=1)
    args = parser.parse_args()
    return args


def load_defense(args):
    """Single defense"""
    # defense = DEFENSES['Gaussian'].as_randomized()

    """Randomized ensemble of all"""
    defense = Ensemble(
        randomized=True,
        preprocessors=[DEFENSES[p].as_randomized() for p in args.defenses],
        k=args.k,
    )

    """Manually specified ensemble"""
    # defense = Ensemble(
    #     randomized=False,
    #     preprocessors=[
    #         DEFENSES['Gaussian'].as_fixed(kernel_size=(5, 5), sigma=(1, 1)),
    #         # DEFENSES['Median'].as_fixed(kernel_size=(5, 5)),
    #         # DEFENSES['JpegCompression'].as_fixed(quality=60),
    #     ],
    #     k=3,
    # )
    return defense


# noinspection DuplicatedCode
def main(args):
    # Basic
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
    if args.norm == 'inf':
        args.eps /= 255
        args.lr /= 255
    else:
        args.norm = int(args.norm)

    # Load data
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), lambda x: x.numpy().astype(np.float32)])
    dataset = ImageFolder(args.data_dir, transform=transform)
    subset = [dataset[i] for i in trange(0, len(dataset), args.data_skip, desc='Load dataset')]
    x_test = np.stack([x for x, y in subset])
    y_test = np.array([y for x, y in subset])
    logger.debug(f'Loaded dataset x: {x_test.shape}, y: {y_test.shape}.')

    # Load model
    model = nn.Sequential(
        NormalizationLayer.preset('imagenet'),
        resnet18(pretrained=True),
    )
    logger.debug(f'Loaded model from "pretrained".')

    # Load attack
    logger.debug(f'Attack: norm {args.norm}, eps {args.eps:.5f}, eps_step {args.lr:.5f}, step {args.step}')
    attack_fn = partial(PGD, norm=args.norm, eps=args.eps, eps_step=args.lr, max_iter=args.step)

    # Load defense
    defense = load_defense(args)
    logger.debug(f'Defense: {defense}.')

    # Load test
    testkit = TestKit(model, defense, attack_fn, args.batch, args.n)

    # 1. Test benign
    preds_clean, acc = testkit.predict(x_test, y_test)
    logger.info(f'Accuracy: {acc:.2f}')

    # 2. Select correctly classified samples
    indices = np.nonzero(preds_clean)
    x_test = x_test[indices]
    y_test = y_test[indices]

    # 3. Test adversarial (non-adaptive)
    preds_adv, rob = testkit.attack(x_test, y_test, adaptive=False, eot_samples=1)
    logger.info(f'Robustness (non-adaptive): {rob:.2f}')

    # 4. Test adversarial (adaptive)
    preds_adv, rob = testkit.attack(x_test, y_test, adaptive=True, eot_samples=args.eot)
    logger.info(f'Robustness (adaptive): {rob:.2f}')


if __name__ == '__main__':
    main(parse_args())
