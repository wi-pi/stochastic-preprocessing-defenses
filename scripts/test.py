import argparse

import numpy as np
import torch.nn as nn
from art.attacks.evasion import ProjectedGradientDescentPyTorch as PGD
from art.estimators.classification import PyTorchClassifier
from torchvision.datasets import CIFAR10

from src.models import CIFAR10ResNet


def parse_args():
    parser = argparse.ArgumentParser()
    # basic
    parser.add_argument('--load', type=str, default='static/logs/version_0/checkpoints/epoch38-acc0.929.ckpt')
    parser.add_argument('--data-dir', type=str, default='static/datasets')
    parser.add_argument('-b', '--batch', type=int, default=512)
    # attack
    parser.add_argument('--eps', type=float, default=8 / 255)
    parser.add_argument('--lr', type=float, default=1 / 255)
    parser.add_argument('--step', type=int, default=20)
    args = parser.parse_args()
    return args


def main(args):
    # Load test data
    dataset = CIFAR10(args.data_dir, train=False)
    x_test = np.array(dataset.data / 255, dtype=np.float32).transpose((0, 3, 1, 2))  # to channel first
    y_test = np.array(dataset.targets, dtype=np.int)

    # Load model
    model = CIFAR10ResNet.load_from_checkpoint(args.load)
    classifier = PyTorchClassifier(
        model,
        loss=nn.CrossEntropyLoss(),
        input_shape=(3, 32, 32),
        nb_classes=10,
    )

    # Test benign
    preds = classifier.predict(x_test, batch_size=args.batch_size)
    acc = np.mean(preds.argmax(1) == y_test)
    print('accuracy', acc)

    # Load attack
    attack = PGD(classifier, norm=np.inf, eps=args.eps, eps_step=args.lr, max_iter=args.step, batch_size=args.batch)

    # Test adversarial
    x_adv = attack.generate(x_test, y_test)
    preds = classifier.predict(x_adv, batch_size=args.batch_size)
    acc = np.mean(preds.argmax(1) == y_test)
    print('robustness', acc)


if __name__ == '__main__':
    main(parse_args())
