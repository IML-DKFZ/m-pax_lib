import os
import requests
import zipfile

import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

from src.utils.download_url import *


def make_weights_for_balanced_classes(images, nclasses):
    """Computing class weights for balanced sampling.
    From: https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3

    Parameters
    ----------
    images : list
        List with image and respective label.
    nclasses : int
        Number of classes.

    Returns
    -------
    list
        List with weights per class.
    """
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.0] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / (float(count[i]) + 0.0001)
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


class OCTDataModule(pl.LightningDataModule):
    """Datamodule for the OCT dataset, inherits from LightningDataModule.
    The datamodule implements three core methods:
        - prepare_data
        - setup
        - dataloders
    Dataloaders are divided into the repective dataset splits and models.
    """

    def __init__(self, batch_size, resize, data_dir, num_workers, pin_memory, seed):
        """Called upon initialization.

        Parameters
        ----------
        batch_size : int
            Number of observations per batch.
        resize : int
            Quadratical size images are resized to.
        data_dir : string
            Location of data directory.
        num_workers : int
            Number of workers simultaneously loading data.
        pin_memory : bool
            If True loaded data tensors will be put into CUDA pinned memory automatically.
        seed : int
            Selected seed for the RNG in all devices.
        """

        super().__init__()
        self.name = "OCTDataModule"

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.resize = resize
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed

    def prepare_data(self):
        """If dataset path does not exist, the method downloads, extracts,
        and removes compressed file of the dataset.

        """

        if not os.path.exists(os.path.join(self.data_dir, "OCT/test/")):
            print("Downloading and extracting OCT retina data...")

            data_url = "https://polybox.ethz.ch/index.php/s/kcRvwssD2yWGfsN/download"
            save_path = os.path.join(self.data_dir, "OCT/download_file.zip")

            os.makedirs(os.path.join(self.data_dir, "OCT/"), exist_ok=True)

            download_url(data_url, save_path)

            zip_ref = zipfile.ZipFile(save_path, "r")
            zip_ref.extractall(self.data_dir + "/OCT/")
            zip_ref.close()

            os.remove(save_path)

    def setup(self):
        """Initializes dataset, randomly splits it and computes weights for weighted sampling."""
        transform_img = transforms.Compose(
            [
                transforms.Resize((self.resize, self.resize)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ]
        )

        train = datasets.ImageFolder(
            self.data_dir + "/OCT/train", transform=transform_img
        )

        data_enc, self.train_head, self.val_head = random_split(
            train,
            [106000, 1700, 609],
            generator=torch.Generator().manual_seed(self.seed),
        )

        self.train_enc, self.val_enc = random_split(
            data_enc, [84600, 21400], generator=torch.Generator().manual_seed(self.seed)
        )

        self.test = datasets.ImageFolder(
            self.data_dir + "/OCT/test", transform=transform_img
        )

        # Computing and distributing weights for weighted sampling
        weights = make_weights_for_balanced_classes(train.imgs, len(train.classes))
        weights = torch.DoubleTensor(weights)

        weights_enc, weights_head_train, weights_head_val = random_split(
            weights,
            [106000, 1700, 609],
            generator=torch.Generator().manual_seed(self.seed),
        )

        weights_enc, _ = random_split(
            weights_enc,
            [84600, 21400],
            generator=torch.Generator().manual_seed(self.seed),
        )

        self.sampler_head_train = torch.utils.data.sampler.WeightedRandomSampler(
            weights_head_train, len(weights_head_train)
        )
        self.sampler_head_val = torch.utils.data.sampler.WeightedRandomSampler(
            weights_head_val, len(weights_head_val)
        )

        self.sampler_enc = torch.utils.data.sampler.WeightedRandomSampler(
            weights_enc, len(weights_enc)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_enc,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def train_dataloader_head(self):
        return DataLoader(
            self.train_head,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.sampler_head_train,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_enc,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader_head(self):
        return DataLoader(
            self.val_head,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.sampler_head_val,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
