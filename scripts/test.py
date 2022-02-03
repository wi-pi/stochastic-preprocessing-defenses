import argparse
import os

import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from art.attacks.evasion import ProjectedGradientDescentPyTorch as PGD
from art.estimators.classification import PyTorchClassifier
from torchvision.datasets import CIFAR10

from src.defenses import EOT, RandomPickOne
from src.models import CIFAR10ResNet

DEFENSES = RandomPickOne([
    T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
    T.RandomRotation(degrees=5),
    T.RandomResizedCrop(size=(32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    T.RandomRotation(degrees=5),
    T.GaussianBlur(kernel_size=5, sigma=(0.01, 0.1)),
])


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
    parser.add_argument('--eot', type=int, default=1)
    args = parser.parse_args()
    return args


def get_wrapper(model: nn.Module, nb_samples: int):
    wrapper = PyTorchClassifier(
        model, loss=nn.CrossEntropyLoss(), input_shape=(3, 32, 32), nb_classes=10,
        preprocessing_defences=EOT(transform=DEFENSES, nb_samples=nb_samples)
    )
    return wrapper


def main(args):
    # Basic
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'

    # Load test data
    dataset = CIFAR10(args.data_dir, train=False)
    x_test = np.array(dataset.data / 255, dtype=np.float32).transpose((0, 3, 1, 2))  # to channel first
    y_test = np.array(dataset.targets, dtype=np.int)

    # Load model
    model = CIFAR10ResNet.load_from_checkpoint(args.load)
    model_predict = get_wrapper(model, nb_samples=1)
    model_attack = get_wrapper(model, nb_samples=args.eot)

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
