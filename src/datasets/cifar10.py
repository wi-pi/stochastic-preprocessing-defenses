import torchvision.transforms as T
from torchvision.datasets import CIFAR10

from src.datasets.base import AbstractBaseDataModule


class CIFAR10DataModule(AbstractBaseDataModule):
    """
    Data module for CIFAR-10.
    """
    transform_train = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()])
    transform_test = T.ToTensor()

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: str = None):
        # full training data
        ds_train_noisy = CIFAR10(self.data_dir, train=True, transform=self.transform_train)
        ds_train_normal = CIFAR10(self.data_dir, train=True, transform=self.transform_test)

        # split train / val / test
        self.ds_train, self.ds_val = self.split_train_val(train=ds_train_noisy, val=ds_train_normal, seed=0)
        self.ds_test = CIFAR10(self.data_dir, train=False, transform=self.transform_test)
