import argparse
import os
from pathlib import Path

import eagerpy as ep
import foolbox as fb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchmetrics as tm
import torchvision.transforms as T
from loguru import logger
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from tqdm import tqdm, trange

from src.utils.gpu import setgpu
from . import resnet


class Test(object):

    def __init__(self, model: nn.Module, bounds: tuple[int, int] = (0, 1)):
        self.model = model
        self.fb_model = fb.models.PyTorchModel(model, bounds=(0, 1))
        self.bounds = bounds

    def test_benign(self, data_loader: DataLoader):
        metric = tm.Accuracy(num_classes=10)
        for x_test, y_test in tqdm(data_loader, desc='Prediction', leave=False):
            preds = self.model(x_test.cuda()).softmax(dim=-1).cpu()
            metric(preds, y_test)

        logger.info(f'Accuracy (benign): {metric.compute() * 100:.2f}')

    def test_attack(self, data_loader: DataLoader, *, eps: float, lr: float, step: int, eot: int):
        metric = tm.Accuracy(num_classes=10)
        acc_all, cnt_all = [], 0
        with tqdm(data_loader, desc='Attack', leave=False) as pbar:
            for x_test, y_test in pbar:
                x_adv, acc_list = self.gradient_estimator_pgd(x_test, y_test, eps=eps, lr=lr, step=step, eot=eot)
                preds = self.model(x_adv).softmax(dim=-1).cpu()
                acc = metric(preds, y_test).item()
                pbar.set_postfix({'adv_acc': f'{acc * 100:.2f}'})
                acc_all.append(acc_list)
                cnt_all += len(x_test)

        logger.info(f'Accuracy (adversarial): {metric.compute() * 100:.2f}')

        acc_all = np.array(acc_all).sum(0) / cnt_all
        return acc_all

    def best_other_classes(self, logits: ep.Tensor, exclude: ep.Tensor) -> ep.Tensor:
        other_logits = logits - ep.onehot_like(logits, exclude, value=ep.inf)
        return other_logits.argmax(axis=-1)

    def loss_fn(self, x, classes):
        logits = self.fb_model(x)

        c_minimize = classes
        c_maximize = self.best_other_classes(logits, classes)

        N = len(x)
        rows = range(N)

        logits_diffs = logits[rows, c_minimize] - logits[rows, c_maximize]
        assert logits_diffs.shape == (N,)

        return logits_diffs

    def es_gradient_estimator(self, x, y, samples, sigma, clip=False):
        gradient = ep.zeros_like(x)
        for k in range(samples // 2):
            noise = ep.normal(x, shape=x.shape)

            pos_theta = x + sigma * noise
            neg_theta = x - sigma * noise

            if clip:
                pos_theta = pos_theta.clip(*self.bounds)
                neg_theta = neg_theta.clip(*self.bounds)

            pos_loss = self.loss_fn(pos_theta, y)
            neg_loss = self.loss_fn(neg_theta, y)

            gradient += (pos_loss - neg_loss)[:, None, None, None] * noise

        gradient /= 2 * sigma * 2 * samples

        return gradient

    def gradient_estimator_pgd(self, x, y, *, eps: float, lr: float, step: int, eot: int):
        """
        Run attack.

        Previous attack choose [100, 1k, 20k] before iterations [20, 40, end].

        :param x: Test images.
        :param y: Test labels.
        :param eps: Perturbation budget.
        :param lr: PGD step size.
        :param step: PGD iterations.
        :param eot: EOT samples.
        :return:
        """
        ep_images = ep.astensor(x.cuda())
        ep_labels = ep.astensor(y.cuda())
        deltas = ep.zeros_like(ep_images)
        mask = self.loss_fn(ep_images, ep_labels) >= 0

        acc_history = []

        for it in trange(step, desc='PGD', leave=False):
            pert_images = (ep_images + deltas).clip(0, 1)
            grads = self.es_gradient_estimator(pert_images[mask], ep_labels[mask], eot, eps)

            # update only subportion of deltas
            # _deltas = np.array(deltas.numpy())
            # _deltas[mask.numpy()] = (deltas[mask] - lr * grads.sign()).numpy()
            # deltas = ep.from_numpy(deltas, _deltas)
            deltas.raw[mask.raw] = deltas.raw[mask.raw] - lr * grads.raw.sign()

            deltas = deltas.clip(-eps, eps)
            pert_images = (ep_images + deltas).clip(0, 1)

            new_logit_diffs = self.loss_fn(pert_images, ep_labels)
            mask = new_logit_diffs >= 0

            if (it + 1) % 10 == 0:
                acc_history.append(mask.sum().item())

            # if mask.sum() == 0:
            #     break

        return pert_images.raw, acc_history


# noinspection DuplicatedCode
def parse_args():
    parser = argparse.ArgumentParser()
    # basic
    parser.add_argument('-b', '--batch', type=int, default=700)
    parser.add_argument('-g', '--gpu', type=int)
    # attack
    parser.add_argument('--eps', type=float, default=8)
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--step', type=int, default=10)
    parser.add_argument('--eot', type=int, default=1)
    # model & dataset
    parser.add_argument('--model-dir', type=Path, default='static/models')
    parser.add_argument('--data-dir', type=Path, default='static/datasets')
    parser.add_argument('--data-skip', type=int, default=15)
    parser.add_argument('--output', type=Path, default='static/logs/kwta')
    args = parser.parse_args()
    return args


# noinspection DuplicatedCode
def main(args):
    # Basic
    setgpu(args.gpu, gb=10.0)
    eps = args.eps / 255
    lr = args.lr / 255
    eot = args.eot * 2  # due to double-side differentiation
    os.makedirs(args.output, exist_ok=True)

    # Load data
    dataset = CIFAR10(args.data_dir, train=False, transform=T.ToTensor())
    dataset = Subset(dataset, indices=list(range(0, len(dataset), args.data_skip)))
    loader = DataLoader(dataset, shuffle=False, batch_size=args.batch, num_workers=16)

    # Load model
    gamma = 0.1
    model = resnet.SparseResNet18(sparsities=[gamma, gamma, gamma, gamma], sparse_func='vol').cuda()
    model.load_state_dict(torch.load(args.model_dir / f'kwta_spresnet18_{gamma}_cifar_adv.pth'))
    model.eval()

    # Load test kit
    test = Test(model)
    test.test_benign(loader)
    acc_list = test.test_attack(loader, eps=eps, lr=lr, step=args.step, eot=eot)

    # Save
    df = pd.DataFrame({
        'Accuracy': acc_list,
        'PGD Steps': np.arange(10, args.step + 1, 10),
    })
    df.to_csv(args.output / f'{args.eps:.1f}_{args.lr:.2f}_{args.step}_{args.eot}.csv', index=False)


if __name__ == '__main__':
    main(parse_args())
