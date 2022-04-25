import abc
from pathlib import Path
from typing import Callable

import numpy as np
import pytorch_lightning as pl
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, Subset

from src.defenses import InstancePreprocessorPyTorch


def identity(x):
    return x


class AbstractBaseDataModule(pl.LightningDataModule, abc.ABC):
    """
    Abstract base datamodule.
    """

    # set attributes
    ds_train: Dataset
    ds_val: Dataset
    ds_test: Dataset

    # default transformations
    transform_train: Callable = NotImplemented
    transform_test: Callable = NotImplemented

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        defense: InstancePreprocessorPyTorch | None,
        seed: int,
        test_batch_multiply: int = 4,
        val_ratio: float = 0.1,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.test_batch_multiply = test_batch_multiply
        self.val_ratio = val_ratio

        defense_transform = defense.forward_one if defense else identity
        self.transform_train = T.Compose([self.transform_train, defense_transform])

    def prepare_data(self):
        pass

    @abc.abstractmethod
    def setup(self, stage: str = None):
        raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(self.ds_train, self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.ds_val, self.batch_size * self.test_batch_multiply, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.ds_test, self.batch_size * self.test_batch_multiply, num_workers=self.num_workers)

    def split_train_val(self, train: Dataset, val: Dataset, seed: int = 0) -> tuple[Dataset, Dataset]:
        """
        Randomly split multiple same-sized datasets into train / val subsets.

        Examples:
            Set 1: 0 1 2 3 4 5 6 7
                   ^^^^^^^^^^^.... train
            Set 2: 0 1 2 3 4 5 6 7
                   ............^^^ val
        """

        # check sizes
        if len(train) != len(val):
            raise ValueError('Datasets should have the same size.')

        # calculate size & indices
        nb_total = len(train)
        nb_val = int(nb_total * self.val_ratio)
        indices = np.random.RandomState(seed).permutation(nb_total)

        # split
        train = Subset(train, indices[:-nb_val])
        val = Subset(val, indices[-nb_val:])

        return train, val
