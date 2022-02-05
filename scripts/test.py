import argparse
import os
from typing import Optional

import numpy as np
import torch.nn as nn
from art.attacks.evasion import ProjectedGradientDescentPyTorch as PGD
from art.defences.preprocessor.preprocessor import PreprocessorPyTorch
from torchvision.datasets import CIFAR10

from src.art.classifier import PyTorchClassifier
from src.defenses.base import DEFENSES
from src.models import CIFAR10ResNet


def parse_args():
    parser = argparse.ArgumentParser()
    # basic
    parser.add_argument('--load', type=str, default='static/logs/version_0/checkpoints/epoch38-acc0.929.ckpt')
    parser.add_argument('--data-dir', type=str, default='static/datasets')
    parser.add_argument('-b', '--batch', type=int, default=512)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    # attack
    parser.add_argument('--eps', type=float, default=8 / 255)
    parser.add_argument('--lr', type=float, default=2 / 255)
    parser.add_argument('--step', type=int, default=10)
    parser.add_argument('--adaptive', action='store_true')
    # defense
    parser.add_argument('-d', '--defense', type=str, choices=DEFENSES, required=True)
    args = parser.parse_args()
    return args


def get_wrapper(model: nn.Module, defense: Optional[PreprocessorPyTorch] = None):
    wrapper = PyTorchClassifier(
        model, loss=nn.CrossEntropyLoss(), input_shape=(3, 32, 32), nb_classes=10, clip_values=(0, 1),
        preprocessing_defences=defense,
    )
    return wrapper


def main(args):
    # Basic
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'

    # Load test data
    dataset = CIFAR10(args.data_dir, train=False)
    x_test = np.array(dataset.data / 255, dtype=np.float32).transpose((0, 3, 1, 2))  # to channel first
    y_test = np.array(dataset.targets, dtype=np.int)

    # Load defense
    defense_cls = DEFENSES[args.defense]
    defense = defense_cls()
    print('using defense', defense)

    # Load model
    model = CIFAR10ResNet.load_from_checkpoint(args.load)
    model_predict = get_wrapper(model, defense=defense)
    model_attack = get_wrapper(model, defense=defense if args.adaptive else None)

    # Test benign
    preds = model_predict.predict(x_test, batch_size=args.batch)
    acc = np.mean(preds.argmax(1) == y_test)
    print('accuracy', acc)

    # Load attack
    attack = PGD(model_attack, norm=np.inf, eps=args.eps, eps_step=args.lr, max_iter=args.step, batch_size=args.batch)

    # Test adversarial
    x_adv = attack.generate(x_test, y_test)
    preds = model_predict.predict(x_adv, batch_size=args.batch)
    acc = np.mean(preds.argmax(1) == y_test)
    print('robustness', acc)


if __name__ == '__main__':
    main(parse_args())
