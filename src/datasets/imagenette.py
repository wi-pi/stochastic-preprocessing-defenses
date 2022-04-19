import torchvision.transforms as T
from torchvision.datasets import ImageFolder

from src.datasets.base import AbstractBaseDataModule


class ImageNetteDataModule(AbstractBaseDataModule):
    """
    Data module for ImageNette.
    """
    transform_train = T.Compose([T.RandomCrop(224), T.RandomHorizontalFlip(), T.ToTensor()])
    transform_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

    def setup(self, stage: str = None):
        # full training data
        ds_train_noisy = ImageFolder(self.data_dir / 'train', transform=self.transform_train)
        ds_train_normal = ImageFolder(self.data_dir / 'train', transform=self.transform_test)

        # split train / val / test
        self.ds_train, self.ds_val = self.split_train_val(train=ds_train_noisy, val=ds_train_normal, seed=0)
        self.ds_test = ImageFolder(self.data_dir / 'val', transform=self.transform_test)
