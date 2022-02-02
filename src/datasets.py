import pytorch_lightning as pl
import torch

import torchvision.transforms as T
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision.datasets import CIFAR10


class CIFAR10DataModule(pl.LightningDataModule):
    ds_train: Dataset
    ds_val: Dataset
    ds_test: Dataset

    def __init__(self, data_dir: str, batch_size: int, num_workers: int, seed: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]
        )

        self.test_transform = T.Compose([
            T.ToTensor(),
        ])

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: str = None):
        # full training data
        ds_train = CIFAR10(self.data_dir, train=True, transform=self.train_transform)
        ds_val = CIFAR10(self.data_dir, train=True, transform=self.test_transform)

        # split train / val
        self.ds_train, _ = random_split(ds_train, [45000, 5000], torch.Generator().manual_seed(self.seed))
        _, self.ds_val = random_split(ds_val, [45000, 5000], torch.Generator().manual_seed(self.seed))

        # split test
        self.ds_test = CIFAR10(self.data_dir, train=False, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(self.ds_val, self.batch_size * 4, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.ds_test, self.batch_size * 4, num_workers=self.num_workers)
