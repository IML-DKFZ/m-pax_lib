import os
import requests
import zipfile

import numpy as np
import pickle5 as pickle
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms

from src.utils.download_url import *


class DiagVibSixDataset(Dataset):
    """Dataset class for DiagViB-6 dataset, inherits from Dataset."""

    def __init__(self, images, labels, transform=None):
        """Called upon initialization.

        Parameters
        ----------
        images : list
            List containing DiagViB-6 images.
        labels : list
            List containing DiagViB labels.
        transform : torchvision.transforms, optional
            Transformation pipeline applied to images, by default None.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """Returns observation from dataset.
        Changes the digits two and five to one and two.

        Parameters
        ----------
        index : tensor
            Index of observation.

        Returns
        -------
        torch.Tensor, int
            Returns image and label.
        """
        if torch.is_tensor(index):
            index = index.tolist()

        images = self.images[index]

        labels = self.labels[index]

        if labels == 2:
            labels = 1
        if labels == 5:
            labels = 2

        if self.transform:
            images = self.transform(images)
            images = images.permute(1, 2, 0)

        return images, labels

    def __len__(self):
        """Returns the dataset size.

        Returns
        -------
        int
            Length of images.
        """
        return len(self.images)


class DiagVibSixDataModule(pl.LightningDataModule):
    """Datamodule for the DiagViB-6 dataset, inherits from LightningDataModule.
    The datamodule implements three core methods:
        - prepare_data
        - setup
        - dataloders
    Dataloaders are divided into the repective dataset splits and models.
    """

    def __init__(
        self, batch_size, resize, data_dir, study, num_workers, pin_memory, seed
    ):
        """Called upon initialization.

        Parameters
        ----------
        batch_size : int
            Number of observations per batch.
        resize : int
            Quadratical size images are resized to.
        data_dir : string
            Location of data directory.
        study : string
            Selected study.
        num_workers : int
            Number of workers simultaneously loading data.
        pin_memory : bool
            If True loaded data tensors will be put into CUDA pinned memory automatically.
        seed : int
            Selected seed for the RNG in all devices.
        """
        super().__init__()
        self.name = "DiagVibSixDataModule"

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.resize = resize
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.study = study

    def prepare_data(self):
        """If dataset path does not exist, the method downloads, extracts,
        and removes compressed file of the dataset.

        """
        if not os.path.exists(os.path.join(self.data_dir, "DiagVibSix/ZGO/")):
            data_url = "https://polybox.ethz.ch/index.php/s/kiBtDsesSzegXMz/download"
            save_path = os.path.join(self.data_dir, "DiagVibSix/download_file.zip")

            os.makedirs(os.path.join(self.data_dir, "DiagVibSix/"), exist_ok=True)

            print("Downloading and extracting DiagViB-6 data...")

            download_url(data_url, save_path)

            zip_ref = zipfile.ZipFile(save_path, "r")
            zip_ref.extractall(self.data_dir + "/DiagVibSix/")
            zip_ref.close()

            os.remove(save_path)

    def setup(self):
        """Opens all three pickeled datasets, loads the dataloaders,
        and randomly splits training and validation data for encoder and head.

        """
        transform_img = transforms.Compose([transforms.ToTensor()])

        file = open(self.data_dir + "/DiagVibSix/" + self.study + "/train.pkl", "rb")
        train_data = pickle.load(file)
        file.close()

        self.train = DiagVibSixDataset(
            train_data["images"],
            train_data["task_labels"],
            transform=transform_img,
        )

        file = open(self.data_dir + "/DiagVibSix/" + self.study + "/val.pkl", "rb")
        val_data = pickle.load(file)
        file.close()

        self.val = DiagVibSixDataset(
            val_data["images"],
            val_data["task_labels"],
            transform=transform_img,
        )

        file = open(self.data_dir + "/DiagVibSix/" + self.study + "/test.pkl", "rb")
        test_data = pickle.load(file)
        file.close()

        self.test = DiagVibSixDataset(
            test_data["images"],
            test_data["task_labels"],
            transform=transform_img,
        )

        self.train_enc, self.train_head = random_split(
            self.train,
            # Same size dataset for head training (740)
            [len(train_data["images"]) - 740, 740],
            generator=torch.Generator().manual_seed(self.seed),
        )

        self.val_enc, self.val_head = random_split(
            self.val,
            # Same size dataset for head validation (648)
            [len(val_data["images"]) - 648, 648],
            generator=torch.Generator().manual_seed(self.seed),
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
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
