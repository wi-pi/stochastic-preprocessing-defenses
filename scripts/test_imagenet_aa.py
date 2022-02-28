import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from autoattack import AutoAttack
from autoattack.autopgd_base import APGDAttack, APGDAttack_targeted
from loguru import logger
from torchvision import models

from configs import DEFENSES, load_defense
from src.defenses import InstancePreprocessorPyTorch
from src.models.layers import NormalizationLayer
from src.utils.gpu import setgpu

# https://pytorch.org/vision/stable/models.html
PRETRAINED_MODELS = {
    'r18': models.resnet18,  # acc = 69.50
    'r50': models.resnet50,  # acc = 75.92
    'inception': models.inception_v3,  # acc = 77.18
}


class DefenseWrapper(nn.Module):

    def __init__(self, defense: InstancePreprocessorPyTorch):
        super().__init__()
        self.defense = defense

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.defense.forward(x, None)
        return x


class APGDAttack_OneTargeted(APGDAttack_targeted):
    # FIXME: Work in progress.

    def perturb(self, x, y=None, x_init=None):
        assert self.loss in ['dlr-targeted']  # 'ce-targeted'
        assert not self.is_tf_model
        if not y is None and len(y.shape) == 0:
            x.unsqueeze_(0)
            y.unsqueeze_(0)
        self.init_hyperparam(x)

        x = x.detach().clone().float().to(self.device)
        y_pred = self.model(x).max(1)[1]
        if y is None:
            # y_pred = self._get_predicted_label(x)
            y = y_pred.detach().clone().long().to(self.device)
        else:
            y = y.detach().clone().long().to(self.device)

        adv = x.clone()
        acc = y_pred == y
        if self.verbose:
            print('-------------------------- ',
                  'running {}-attack with epsilon {:.5f}'.format(
                      self.norm, self.eps),
                  '--------------------------')
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))

        startt = time.time()

        torch.random.manual_seed(self.seed)
        torch.cuda.random.manual_seed(self.seed)

        target_class = self.n_target_classes
        for counter in range(self.n_restarts):
            ind_to_fool = acc.nonzero().squeeze()
            if len(ind_to_fool.shape) == 0:
                ind_to_fool = ind_to_fool.unsqueeze(0)
            if ind_to_fool.numel() != 0:
                x_to_fool = x[ind_to_fool].clone()
                y_to_fool = y[ind_to_fool].clone()

                if not self.is_tf_model:
                    output = self.model(x_to_fool)
                else:
                    output = self.model.predict(x_to_fool)
                self.y_target = output.sort(dim=1)[1][:, target_class]

                res_curr = self.attack_single_run(x_to_fool, y_to_fool)
                best_curr, acc_curr, loss_curr, adv_curr = res_curr
                ind_curr = (acc_curr == 0).nonzero().squeeze()

                acc[ind_to_fool[ind_curr]] = 0
                adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                if self.verbose:
                    print('target class {}'.format(target_class),
                          '- restart {} - robust accuracy: {:.2%}'.format(
                              counter, acc.float().mean()),
                          '- cum. time: {:.1f} s'.format(
                              time.time() - startt))

        return adv


def parse_args():
    parser = argparse.ArgumentParser()
    # basic
    parser.add_argument('--load', type=str, default='r18', choices=PRETRAINED_MODELS)
    parser.add_argument('-b', '--batch', type=int, default=250)
    parser.add_argument('-g', '--gpu', type=int)
    parser.add_argument('-n', type=int, default=10)
    # dataset
    parser.add_argument('--data-dir', type=str, default='static/datasets/imagenet')
    parser.add_argument('--data-skip', type=int, default=50)
    # attack
    parser.add_argument('--norm', type=str, default='inf')
    parser.add_argument('--eps', type=float, default=8)
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--step', type=int, default=10)
    parser.add_argument('-t', '--target', type=int, default=-1)
    # defense
    parser.add_argument('-d', '--defenses', type=str, choices=DEFENSES, nargs='+')
    parser.add_argument('-k', type=int)
    parser.add_argument('--eot', type=int, default=1)
    args = parser.parse_args()
    return args


# noinspection DuplicatedCode
def main(args):
    # Basic
    setgpu(args.gpu, gb=10.0)
    targeted = args.target >= 0
    if args.norm == 'inf':
        args.eps /= 255
        args.lr /= 255
    else:
        args.norm = int(args.norm)

    # Load data
    subset = np.load(f'./static/imagenet.{args.data_skip}.npz')
    x_test = subset['x_test']
    y_test = subset['y_test']
    logger.debug(f'Loaded dataset x: {x_test.shape}, y: {y_test.shape}.')

    # Load defense
    defense = load_defense(args.defenses)
    logger.debug(f'Defense: {defense}.')

    # Load model
    model = nn.Sequential(
        DefenseWrapper(defense),
        NormalizationLayer.preset('imagenet'),
        PRETRAINED_MODELS[args.load](pretrained=True),
    ).cuda()
    logger.debug(f'Loaded model from "{args.load}".')

    # Load attack
    attack = AutoAttack(model, norm='Linf', eps=args.eps, version='rand')
    if targeted:
        logger.debug(f'Test with target {args.target}')
        attack.attacks_to_run = ['apgd-t']
        attack.apgd_targeted = APGDAttack_OneTargeted(
            model,
            norm='Linf',
            eps=args.eps,
            n_iter=args.step,
            eot_iter=args.eot,
            n_target_classes=args.target,
            n_restarts=1,
            verbose=True,
            rho=.75,
            device='cuda',
        )
    else:
        logger.debug(f'Test with no target.')
        attack.attacks_to_run = ['apgd-ce']
        attack.apgd = APGDAttack(
            model,
            norm='Linf',
            eps=args.eps,
            n_iter=args.step,
            eot_iter=args.eot,
            n_restarts=1,
            verbose=True,
            rho=.75,
            device='cuda',
        )

    # Eval
    with torch.no_grad():
        x_test, y_test = map(torch.from_numpy, [x_test, y_test])
        x_adv, y_adv = attack.run_standard_evaluation(x_test, y_test, bs=args.batch, return_labels=True)


if __name__ == '__main__':
    main(parse_args())
