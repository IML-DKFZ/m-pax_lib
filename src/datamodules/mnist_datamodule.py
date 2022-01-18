from typing import Optional, Tuple

import torch

from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


class MNISTDataModule(LightningDataModule):
    """Datamodule for the MNIST dataset, inherits from LightningDataModule.
    The datamodule implements three core methods:
        - prepare_data
        - setup
        - dataloders
    Dataloaders are divided into the repective dataset splits and models.
    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = True,
        seed=42,
        resize=32,
    ):
        """Called upon initialization.

        Parameters
        ----------
        data_dir : str, optional
            Location of data directory, by default "data/".
        batch_size : int
            Number of observations per batch, by default 64.
        num_workers : int, optional
            Number of workers simultaneously loading data, by default 0.
        pin_memory : bool, optional
            If True loaded data tensors will be put into CUDA pinned memory automatically, by default False.
        drop_last : bool, optional
            Drop last batch if smaller than batch_size, by default True.
        seed : int, optinal
            Selected seed for the RNG in all devices, by default 42.
        resize : int, optional
            Quadratical size images are resized to, by default 32.
        """
        super().__init__()
        self.name = "MNISTDataModule"

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.seed = seed

        self.transform = transforms.Compose(
            [transforms.Resize(resize), transforms.ToTensor()]
        )

    def prepare_data(self):
        """If dataset path does not exist, the method downloads and extracts dataset.
        This method is called only from a single GPU.

        """
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self):
        """Initializes dataset and randomly splits it."""
        mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)

        data_head, data_enc = random_split(
            mnist_full,
            [1000, 59000],
            generator=torch.Generator().manual_seed(self.seed),
        )

        self.train_enc, self.val_enc = random_split(
            data_enc, [55000, 4000], generator=torch.Generator().manual_seed(self.seed)
        )

        self.train_head, self.val_head = random_split(
            data_head, [800, 200], generator=torch.Generator().manual_seed(self.seed)
        )

        self.test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_enc,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )

    def train_dataloader_head(self):
        return DataLoader(
            self.train_head,
            batch_size=32,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_enc,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )

    def val_dataloader_head(self):
        return DataLoader(
            self.val_head,
            batch_size=32,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )
