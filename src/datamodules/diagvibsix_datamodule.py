import pickle5 as pickle
import numpy as np
import os
import zipfile
import requests

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
import pytorch_lightning as pl

from src.utils.download_url import *

class DiagVibSixDataset(Dataset):
    def __init__(self, images, labels, study, transform=None):
        self.images = images
        self.labels = labels
        self.study = study
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        x = self.images[index]

        y = self.labels[index]

        if y == 2:
            y = 1
        if y == 5:
            y = 2


        if self.transform:
            x = self.transform(x)
            x = x.permute(1, 2, 0)

        return x, y


class DiagVibSixDataModule(pl.LightningDataModule):
    def __init__(
        self, batch_size, resize, data_dir, study, num_workers, pin_memory, seed
    ):
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
        if not os.path.exists(os.path.join(self.data_dir, "DiagVibSix/ZGO/")):
            data_url = "https://polybox.ethz.ch/index.php/s/aEWoWH2yv1HFws4/download"
            save_path = os.path.join(self.data_dir, "DiagVibSix/download_file.zip")

            os.makedirs(os.path.join(self.data_dir, "DiagVibSix/"))

            print("Downloading and extracting DiagVibSix data...")

            download_url(data_url, save_path)

            zip_ref = zipfile.ZipFile(save_path, "r")
            zip_ref.extractall(self.data_dir + "/DiagVibSix/")
            zip_ref.close()

            os.remove(save_path)

    def setup(self):
        transform_img = transforms.Compose([transforms.ToTensor()])

        file = open(self.data_dir + "/DiagVibSix/" + self.study + "/train.pkl", "rb")
        train_data = pickle.load(file)
        file.close()

        self.train = DiagVibSixDataset(
            train_data["images"],
            train_data["task_labels"],
            self.study,
            transform=transform_img,
        )

        file = open(self.data_dir + "/DiagVibSix/" + self.study + "/val.pkl", "rb")
        val_data = pickle.load(file)
        file.close()

        self.val = DiagVibSixDataset(
            val_data["images"],
            val_data["task_labels"],
            self.study,
            transform=transform_img,
        )

        file = open(self.data_dir + "/DiagVibSix/" + self.study + "/test.pkl", "rb")
        test_data = pickle.load(file)
        file.close()

        self.test = DiagVibSixDataset(
            test_data["images"],
            test_data["task_labels"],
            self.study,
            transform=transform_img,
        )

        self.train_enc, self.train_head = random_split(  # 43740 / 39995
            self.train,
            [len(train_data["images"]) - 740, 740],
            generator=torch.Generator().manual_seed(self.seed),
        )

        self.val_enc, self.val_head = random_split(  # 8748 / 8745
            self.val,
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
