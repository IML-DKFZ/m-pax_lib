import os
import requests
import zipfile

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from urllib.request import urlretrieve
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from PIL import Image

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


class ISICDataset(Dataset):
    """Dataset class for ISIC dataset, inherits from Dataset."""

    def __init__(self, img_path: str, transform=None, csv_path=None):
        """Called upon initialization. Reads label-csv from provided path.

        Parameters
        ----------
        img_path : str
            Path to image folder.
        transform : torchvision.transforms, optional
            Transformation pipeline applied to images, by default None.
        csv_path : str, optional
            Path to csv containing labels, by default None.
        """
        self.labels = pd.read_csv(csv_path)
        self.img_path = img_path
        self.transform = transform

    def __getitem__(self, index):
        """Reads image and returns observation from dataset.

        Parameters
        ----------
        index : tensor
            Index of observation.

        Returns
        -------
        troch.Tensor, int
            Returns image and label.
        """
        img_name = os.path.join(self.img_path, f"{self.labels.iloc[index, 0]}.jpg")
        images = Image.open(img_name)
        images = self.transform(images)

        labels = self.labels.iloc[index, 1:]
        labels = np.array([labels])
        labels = labels.reshape(-1, 9).argmax()
        return images, labels

    def __len__(self):
        """Returns the dataset size.

        Returns
        -------
        int
            Length of images.
        """
        return len(self.labels)


class ISICDataModule(pl.LightningDataModule):
    """Datamodule for the ISIC dataset, inherits from LightningDataModule.
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
        self.name = "ISICDataModule"

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.resize = resize
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed

    def prepare_data(self):
        """If dataset path does not exist, the method downloads, extracts,
        and removes compressed image and label files of the dataset.

        """

        if not os.path.exists(
            os.path.join(self.data_dir, "ISIC/ISIC_2019_Training_Input")
        ):
            print("Downloading and extracting ISIC skin cancer data...")
            data_url = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip"
            save_path = os.path.join(self.data_dir, "ISIC/download_file.zip")

            os.makedirs(os.path.join(self.data_dir, "ISIC/"), exist_ok=True)

            download_url(data_url, save_path)

            zip_ref = zipfile.ZipFile(save_path, "r")
            zip_ref.extractall(self.data_dir + "/ISIC/")
            zip_ref.close()

            os.remove(save_path)

        if not os.path.exists(os.path.join(self.data_dir, "ISIC/labels.csv")):
            print("Downloading and extracting ISIC skin cancer labels...")
            data_url = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv"
            save_path = os.path.join(self.data_dir, "ISIC/labels.csv")

            urlretrieve(data_url, save_path)

    def setup(self):
        """Initializes dataset, randomly splits it and computes weights for weighted sampling."""
        transform_img = transforms.Compose(
            [transforms.Resize((self.resize, self.resize)), transforms.ToTensor()]
        )

        data = ISICDataset(
            img_path=self.data_dir + "/ISIC/ISIC_2019_Training_Input",
            transform=transform_img,
            csv_path=self.data_dir + "/ISIC/labels.csv",
        )

        (
            self.train_enc,
            self.val_enc,
            self.train_head,
            self.val_head,
            self.test,
        ) = random_split(
            data,
            [20000, 2818, 432, 81, 2000],
            generator=torch.Generator().manual_seed(self.seed),
        )

        # Computing and distributing weights for weighted sampling
        weights_train_head = torch.DoubleTensor(
            make_weights_for_balanced_classes(self.train_head, 9)
        )
        weights_val_head = torch.DoubleTensor(
            make_weights_for_balanced_classes(self.val_head, 9)
        )

        self.sampler_train_head = torch.utils.data.sampler.WeightedRandomSampler(
            weights_train_head, len(weights_train_head)
        )
        self.sampler_val_head = torch.utils.data.sampler.WeightedRandomSampler(
            weights_val_head, len(weights_val_head)
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
            sampler=self.sampler_train_head,
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
            sampler=self.sampler_val_head,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
