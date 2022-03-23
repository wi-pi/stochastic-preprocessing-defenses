from functools import partial

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.functional import accuracy
from torchvision import models


def load_pl_models(model: nn.Module, weight_file: str):
    state_dict = torch.load(weight_file, map_location='cpu')['state_dict']
    state_dict = {k.removeprefix('model.'): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model


def resnet18(weight_file: str | None = None):
    model = models.resnet18(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    if weight_file:
        load_pl_models(model, weight_file)
    return model


CIFAR10_MODELS = {
    'r18': partial(resnet18, 'static/models/cifar10/resnet18/checkpoints/epoch38-acc0.929.ckpt'),
    'r18.aug3': partial(resnet18, 'static/models/cifar10/augment3/checkpoints/epoch19-acc0.768.ckpt'),
    'r18.aug4': partial(resnet18, 'static/models/cifar10/augment4/checkpoints/epoch18-acc0.752.ckpt'),
}


class CIFAR10ResNet(pl.LightningModule):

    def __init__(self, lr: float = 1e-3, wd: float = 1e-2, max_epochs: int = 40):
        super().__init__()
        self.save_hyperparameters()
        self.model = resnet18()
        self.criterion = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs)
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.model(x)

    def loss(self, batch, mode):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=-1)

        acc = accuracy(preds, y)
        self.log(f'{mode}_loss', loss, prog_bar=True)
        self.log(f'{mode}_acc', acc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.loss(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        self.loss(batch, mode='val')

    def test_step(self, batch, batch_idx):
        self.loss(batch, mode='test')
