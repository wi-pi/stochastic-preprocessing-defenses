import argparse
import os
from pathlib import Path

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.datasets import CIFAR10

from configs import DEFENSES, load_defense
from src.datasets import ImageNet


def parse_args():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'imagenet'], default='imagenet')
    parser.add_argument('--data-dir', type=Path, default='static/datasets')
    parser.add_argument('-i', '--id', type=int, default=0)
    parser.add_argument('-n', type=int, default=1)
    # defense
    parser.add_argument('-d', '--defenses', type=str, choices=DEFENSES, nargs='+', default=[])
    parser.add_argument('-k', '--nb-defenses', type=int)
    parser.add_argument('-p', '--params', nargs='+', help='additional kwargs passed to defenses')
    # output
    parser.add_argument('--save', type=Path, default='static/visualize')
    parser.add_argument('-t', '--tag', type=str, default='test')
    args = parser.parse_args()
    return args


def main(args):
    # Basic
    os.makedirs(args.save, exist_ok=True)

    # Load dataset
    match args.dataset:
        case 'cifar10':
            dataset = CIFAR10(args.data_dir, train=False, transform=T.ToTensor())
        case 'imagenet':
            dataset = ImageNet(args.data_dir / 'imagenet' / 'val', transform='resnet', skip=None, numpy=False)
        case _:
            raise NotImplementedError(args.dataset)

    # Load test data
    x, _ = dataset[args.id]
    assert torch.is_tensor(x)

    # Load defense
    defense = load_defense(defenses=args.defenses, nb_samples=args.nb_defenses, params=args.params)
    print(defense)

    # Save raw data
    F.to_pil_image(x).save(args.save / f'{args.id}_{args.tag}.png')

    # Save processed data
    for i in range(args.n):
        x_processed = defense.forward_one(x.clone())
        F.to_pil_image(x_processed).save(args.save / f'{args.id}_{args.tag}_{i}.png')


if __name__ == '__main__':
    main(parse_args())
