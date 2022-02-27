import argparse
import os

import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.datasets import CIFAR10, ImageFolder

from configs import DEFENSES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'imagenet'], default='imagenet')
    parser.add_argument('--data-dir', type=str, default='static/datasets')
    parser.add_argument('--save', type=str, default='static/visualize')
    parser.add_argument('-i', '--id', type=int, default=0)
    parser.add_argument('-d', '--defense', type=str, choices=DEFENSES, required=True)
    parser.add_argument('-n', type=int, default=1)
    args = parser.parse_args()
    return args


def main(args):
    # Basic
    os.makedirs(args.save, exist_ok=True)

    # Load test data
    if args.dataset == 'cifar10':
        dataset = CIFAR10(args.data_dir, train=False, transform=T.ToTensor())
    elif args.dataset == 'imagenet':
        transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
        dataset = ImageFolder(os.path.join(args.data_dir, 'imagenet'), transform=transform)
    else:
        raise NotImplementedError(args.dataset)

    # Load defense
    defense_cls = DEFENSES[args.defense]
    defense = defense_cls()
    print('using defense', defense)

    # Save raw data
    x, _ = dataset[args.id]
    F.to_pil_image(x).save(os.path.join(args.save, f'{args.id}.png'))

    # Save processed data
    for i in range(args.n):
        x_processed = defense.forward_one(x.clone())
        F.to_pil_image(x_processed).save(os.path.join(args.save, f'{args.id}_{args.defense}_{i}.png'))


if __name__ == '__main__':
    main(parse_args())
