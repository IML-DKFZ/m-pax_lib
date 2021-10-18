import torch
import os
import requests

import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
import pytorch_lightning as pl

import pandas as pd
import numpy as np
from PIL import Image


class CXR8Dataset(Dataset):
    def __init__(self, img_path, transform, csv_path=None):
        self.targets = pd.read_csv(csv_path)
        self.img_path = img_path
        self.transform = transform

    def __getitem__(self, index):
        img_name = os.path.join(self.img_path, f"{self.targets.iloc[index, 0]}")
        img = Image.open(img_name)
        img = self.transform(img)

        targets = self.targets.iloc[index, 1:]
        targets = np.array([targets])
        targets = targets.astype(np.float32).reshape(-1, 14)
        targets = torch.as_tensor(targets, dtype=torch.float).squeeze()
        return img, targets

    def __len__(self):
        return len(self.targets)


class CXR8DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, resize, data_dir, num_workers, pin_memory, seed):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.resize = resize
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed

    def prepare_data(self):
        pass

    def setup(self):
        transform_img = transforms.Compose(
            [
                transforms.Resize((self.resize, self.resize)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ]
        )

        train = CXR8Dataset(
            img_path=self.data_dir + "/CXR8/images/",
            transform=transform_img,
            csv_path=self.data_dir + "/CXR8/train_label.csv",
        )

        val = CXR8Dataset(
            img_path=self.data_dir + "/CXR8/images/",
            transform=transform_img,
            csv_path=self.data_dir + "/CXR8/val_label.csv",
        )

        self.test = CXR8Dataset(
            img_path=self.data_dir + "/CXR8/images/",
            transform=transform_img,
            csv_path=self.data_dir + "/CXR8/test_label.csv",
        )

        self.train_enc, self.train_head = random_split(
            train,
            [74964, 750],  # 75714
            generator=torch.Generator().manual_seed(self.seed),
        )

        self.val_enc, self.val_head = random_split(
            val,
            [10700, 110],  # 10810
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
