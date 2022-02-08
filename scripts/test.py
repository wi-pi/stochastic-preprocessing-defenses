import argparse
import os
from typing import Any, Optional

import numpy as np
import torch.nn as nn
from art.attacks.evasion import ProjectedGradientDescentPyTorch as PGD
from art.defences.preprocessor.preprocessor import PreprocessorPyTorch
from loguru import logger
from torchvision.datasets import CIFAR10
from tqdm import trange

from src.art.classifier import PyTorchClassifier
from src.defenses import DEFENSES, EOT, Ensemble
from src.models import CIFAR10ResNet

PRETRAINED_MODELS = {
    'common': 'static/logs/common/checkpoints/epoch38-acc0.929.ckpt',
    'augment3': 'static/logs/augment3/checkpoints/epoch19-acc0.768.ckpt',
    'augment4': 'static/logs/augment4/checkpoints/epoch18-acc0.752.ckpt',
}


def parse_args():
    parser = argparse.ArgumentParser()
    # basic
    parser.add_argument('--load', type=str, choices=PRETRAINED_MODELS.keys(), default='augment3')
    parser.add_argument('--data-dir', type=str, default='static/datasets')
    parser.add_argument('-b', '--batch', type=int, default=1000)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-n', type=int, default=20)
    # attack
    parser.add_argument('--norm', type=str, default='inf')
    parser.add_argument('--eps', type=float, default=8)
    parser.add_argument('--lr', type=float, default=2)
    parser.add_argument('--step', type=int, default=10)
    parser.add_argument('-a', '--adaptive', action='store_true')
    # defense
    parser.add_argument('-d', '--defenses', type=str, choices=DEFENSES, nargs='+', required=True)
    parser.add_argument('-k', type=int, default=2)
    parser.add_argument('-r', '--randomized', action='store_true')
    parser.add_argument('--eot', type=int, default=1)
    args = parser.parse_args()
    return args


def get_wrapper(model: nn.Module, defense: Optional[PreprocessorPyTorch] = None):
    wrapper = PyTorchClassifier(
        model, loss=nn.CrossEntropyLoss(), input_shape=(3, 32, 32), nb_classes=10, clip_values=(0, 1),
        preprocessing=None, preprocessing_defences=defense,
    )
    return wrapper


def robust_predict(model: PyTorchClassifier, x_test: np.ndarray, y_test: np.ndarray, args: Any):
    correct = np.ones_like(y_test)
    for _ in trange(args.n, desc='Predict'):
        preds = model.predict(x_test, batch_size=args.batch).argmax(1)
        correct &= preds == y_test
    return correct


def main(args):
    # Basic
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
    args.load = PRETRAINED_MODELS[args.load]
    if args.norm == 'inf':
        args.eps /= 255
        args.lr /= 255
    else:
        args.norm = int(args.norm)

    # Load test data
    dataset = CIFAR10(args.data_dir, train=False)
    x_test = np.array(dataset.data / 255, dtype=np.float32).transpose((0, 3, 1, 2))  # to channel first
    y_test = np.array(dataset.targets, dtype=np.int)
    x_test = x_test[::10]
    y_test = y_test[::10]
    logger.debug(f'Loaded dataset x: {x_test.shape}, y: {y_test.shape}.')

    # Load defense
    """Single defense
    """
    # defense = DEFENSES['Gaussian'].as_randomized()
    """Randomized ensemble of all
    """
    defense = Ensemble(
        randomized=args.randomized,
        preprocessors=[DEFENSES[p].as_randomized() for p in args.defenses],
        k=args.k,
    )
    """Manually specified ensemble 
    """
    # defense = Ensemble(
    #     randomized=False,
    #     preprocessors=[
    #         DEFENSES['Gaussian'].as_fixed(kernel_size=(5, 5), sigma=(1, 1)),
    #         # DEFENSES['Median'].as_fixed(kernel_size=(5, 5)),
    #         # DEFENSES['JpegCompression'].as_fixed(quality=60),
    #     ],
    #     k=3,
    # )
    logger.debug(f'Using defense {defense}.')

    if args.eot > 1:
        args.batch = max(1, args.batch // args.eot)
        defense = EOT(defense, nb_samples=args.eot)
        logger.debug(f'Using EOT with {args.eot} samples.')
    else:
        logger.debug(f'No EOT used.')

    # Load model
    model = CIFAR10ResNet.load_from_checkpoint(args.load)
    model_predict = get_wrapper(model, defense=defense)
    model_attack = get_wrapper(model, defense=defense if args.adaptive else None)

    # Test benign
    preds_clean = robust_predict(model_predict, x_test, y_test, args)
    acc = np.mean(preds_clean)
    logger.info(f'Accuracy: {acc:.4f}')

    # Only test valid samples
    indices = np.nonzero(preds_clean)
    x_test = x_test[indices]
    y_test = y_test[indices]
    logger.debug(f'Selecting {len(x_test)} correctly classified samples.')

    # Load attack
    logger.debug(f'Attack: norm {args.norm}, eps {args.eps:.5f}, eps_step {args.lr:.5f}, step {args.step}')
    attack = PGD(model_attack, args.norm, eps=args.eps, eps_step=args.lr, max_iter=args.step, batch_size=args.batch)

    # Test adversarial
    x_adv = attack.generate(x_test, y_test)
    preds_adv = robust_predict(model_predict, x_adv, y_test, args)
    rob = np.mean(preds_adv)
    logger.info(f'Robustness: {rob:.4f}')


if __name__ == '__main__':
    main(parse_args())
