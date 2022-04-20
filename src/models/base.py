import abc

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.functional import accuracy


class AbstractBaseResNet(pl.LightningModule, abc.ABC):
    """
    Abstract base ResNet for classification.

    Criterion: CrossEntropyLoss
    Optimizer: AdamW
    Scheduler: CosineAnnealing
    """

    def __init__(self, lr: float = 1e-3, wd: float = 1e-2, max_epochs: int = 40):
        super().__init__()
        self.save_hyperparameters()
        self.model = self.make_model()
        self.criterion = nn.CrossEntropyLoss()

    @staticmethod
    @abc.abstractmethod
    def make_model() -> nn.Module:
        raise NotImplementedError

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
