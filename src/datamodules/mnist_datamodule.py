from typing import Optional, Tuple
from six.moves import urllib

from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


class MNISTDataModule(LightningDataModule):
    """
    Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transform = transforms.Compose(
            [transforms.Resize(32), transforms.ToTensor()]
        )

        # self.dims is returned when you call datamodule.size()
        self.dims = (1, 28, 28)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 10

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""

        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self):
        mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)

        data_cla, data_enc = random_split(mnist_full, [1000, 59000])

        self.train_enc, self.val_enc = random_split(data_enc, [50000, 9000])

        self.train_cla, self.val_cla = random_split(data_cla, [800, 200])

        self.test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_enc, batch_size=self.batch_size, num_workers=self.num_workers)

    def train_dataloader_head(self):
        return DataLoader(self.train_cla, batch_size=32, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_enc, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader_head(self):
        return DataLoader(self.val_cla, batch_size=32, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)


#data = MNISTDataModule()
#data.setup()
#batch = next(iter(data.train_dataloader()))