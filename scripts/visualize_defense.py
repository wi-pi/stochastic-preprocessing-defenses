import argparse
import os

import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.datasets import CIFAR10

from src.defenses import DEFENSES


def parse_args():
    parser = argparse.ArgumentParser()
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
    dataset = CIFAR10(args.data_dir, train=False)
    x_test = np.array(dataset.data / 255, dtype=np.float32).transpose((0, 3, 1, 2))  # to channel first

    # Load defense
    defense_cls = DEFENSES[args.defense]
    defense = defense_cls.as_randomized()
    print('using defense', defense)

    # Save raw data
    x = x_test[args.id]
    x_t = torch.from_numpy(x)
    F.to_pil_image(x_t).save(os.path.join(args.save, f'{args.id}.png'))

    # Save processed data
    for i in range(args.n):
        x_processed, _ = defense.forward(x_t.clone()[None])
        F.to_pil_image(x_processed[0]).save(os.path.join(args.save, f'{args.id}_{args.defense}_{i}.png'))


if __name__ == '__main__':
    main(parse_args())
