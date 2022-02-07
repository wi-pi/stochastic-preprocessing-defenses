import argparse
import os
from typing import Optional

import numpy as np
import torch.nn as nn
from art.attacks.evasion import ProjectedGradientDescentPyTorch as PGD
from art.defences.preprocessor.preprocessor import PreprocessorPyTorch
from loguru import logger
from torchvision.datasets import CIFAR10

from src.art.classifier import PyTorchClassifier
from src.defenses import DEFENSES, EOT, Ensemble
from src.models import CIFAR10ResNet


def parse_args():
    parser = argparse.ArgumentParser()
    # basic
    parser.add_argument('--load', type=str, default='static/logs/version_0/checkpoints/epoch38-acc0.929.ckpt')
    parser.add_argument('--data-dir', type=str, default='static/datasets')
    parser.add_argument('-b', '--batch', type=int, default=1000)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    # attack
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


def main(args):
    # Basic
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
    # args.eps /= 255
    # args.lr /= 255

    # Load test data
    dataset = CIFAR10(args.data_dir, train=False)
    x_test = np.array(dataset.data / 255, dtype=np.float32).transpose((0, 3, 1, 2))  # to channel first
    y_test = np.array(dataset.targets, dtype=np.int)
    x_test = x_test[::10]
    y_test = y_test[::10]
    logger.debug(f'Loaded dataset x: {x_test.shape}, y: {y_test.shape}.')

    # Load defense
    # defense = Ensemble(
    #     randomized=args.randomized,
    #     preprocessors=[DEFENSES[p].as_randomized() for p in args.defenses],
    #     k=args.k,
    # )
    # defense = Ensemble(
    #     randomized=True,
    #     preprocessors=[
    #         DEFENSES['Gaussian'].as_fixed(kernel_size=(3, 3), sigma=(1, 1)),
    #         DEFENSES['Gaussian'].as_fixed(kernel_size=(5, 5), sigma=(1, 1)),
    #     ],
    #     k=2,
    # )
    defense = DEFENSES['Gaussian'].as_randomized()
    logger.debug(f'Using defense {defense}.')

    if args.eot > 1:
        args.batch = max(1, args.batch // args.eot)
        defense = EOT(defense, nb_samples=args.eot)
        logger.debug(f'Using EOT with {args.eot} samples.')

    # Load model
    model = CIFAR10ResNet.load_from_checkpoint(args.load)
    model_predict = get_wrapper(model, defense=defense)
    model_attack = get_wrapper(model, defense=defense if args.adaptive else None)

    # Test benign
    preds_clean = model_predict.predict(x_test, batch_size=args.batch).argmax(1)
    acc = np.mean(preds_clean == y_test)
    logger.info(f'Accuracy: {acc:.4f}')

    # Only test valid samples
    indices = np.nonzero(preds_clean == y_test)
    x_test = x_test[indices]
    y_test = y_test[indices]
    logger.debug(f'Selecting {len(x_test)} correctly classified samples.')

    # Load attack
    # logger.debug(f'Attack: norm {np.inf}, eps {args.eps:.3f}, eps_step {args.lr:.3f}, step {args.step}')
    # attack = PGD(model_attack, norm=np.inf, eps=args.eps, eps_step=args.lr, max_iter=args.step, batch_size=args.batch)

    logger.debug(f'Attack: norm {2}, eps {args.eps:.3f}, eps_step {args.lr:.3f}, step {args.step}')
    attack = PGD(model_attack, norm=2, eps=args.eps, eps_step=args.lr, max_iter=args.step, batch_size=args.batch)

    # Test adversarial
    x_adv = attack.generate(x_test)
    preds_adv = model_predict.predict(x_adv, batch_size=args.batch).argmax(1)
    rob = np.mean(preds_adv == y_test)
    logger.info(f'Robustness: {rob:.4f}')

    import torch
    import torchvision.transforms.functional as F
    for i in range(10):
        F.to_pil_image(torch.from_numpy(x_adv[i])).save(f't{i}.png')


if __name__ == '__main__':
    main(parse_args())
