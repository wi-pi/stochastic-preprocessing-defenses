from pathlib import Path
from typing import Callable

import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from tqdm import trange

from src.datasets.utils import ToNumpy

default_transformations = {
    'resnet': T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), ToNumpy(np.float32)])
}


def imagenet(root_dir: Path | str, transform: Callable | str | None, skip: int | None) -> Dataset:
    if isinstance(transform, str):
        transform = default_transformations[transform]

    dataset = ImageFolder(root_dir, transform=transform)

    if skip:
        with trange(0, len(dataset), skip, desc='Load dataset', leave=False) as pbar:
            dataset = [dataset[i] for i in pbar]

    return dataset
