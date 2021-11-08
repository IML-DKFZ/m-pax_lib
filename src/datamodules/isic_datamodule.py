import torch
import os
import zipfile
import requests
from urllib.request import urlretrieve

import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
import pytorch_lightning as pl

import pandas as pd
import numpy as np
from PIL import Image


# From: https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
# Computing class weights for balanced sampling
def make_weights_for_balanced_classes(images, nclasses):
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


def download_url(url, save_path):  # Chunk wise downloading to not overuse RAM
    r = requests.get(url, stream=True, allow_redirects=True)
    with open(save_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                f.flush()


class ISICDataset(Dataset):
    def __init__(self, img_path, transform, csv_path=None):
        self.targets = pd.read_csv(csv_path)
        self.img_path = img_path
        self.transform = transform

    def __getitem__(self, index):
        img_name = os.path.join(self.img_path, f"{self.targets.iloc[index, 0]}.jpg")
        img = Image.open(img_name)
        img = self.transform(img)

        targets = self.targets.iloc[index, 1:]
        targets = np.array([targets])
        targets = targets.reshape(-1, 9).argmax()
        return img, targets

    def __len__(self):
        return len(self.targets)


class ISICDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, resize, data_dir, num_workers, pin_memory, seed):
        super().__init__()
        self.name = "ISICDataModule"

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.resize = resize
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed

    def prepare_data(self):
        if not os.path.exists(
            os.path.join(self.data_dir, "ISIC/ISIC_2019_Training_Input")
        ):
            data_url = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip"
            save_path = os.path.join(self.data_dir, "ISIC/download_file.zip")

            os.makedirs(os.path.join(self.data_dir, "ISIC/"), exist_ok=True)

            print("Downloading and extracting ISIC skin cancer data...")

            download_url(data_url, save_path)

            zip_ref = zipfile.ZipFile(save_path, "r")
            zip_ref.extractall(self.data_dir + "/ISIC/")
            zip_ref.close()

            os.remove(save_path)

        if not os.path.exists(os.path.join(self.data_dir, "ISIC/labels.csv")):
            data_url = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv"
            save_path = os.path.join(self.data_dir, "ISIC/labels.csv")

            print("Downloading and extracting ISIC skin cancer labels...")

            urlretrieve(data_url, save_path)

    def setup(self):
        transform_img = transforms.Compose(
            [
                transforms.Resize((self.resize, self.resize)),  # Bilinear resizing
                transforms.ToTensor(),
                # transforms.Normalize((0.6678, 0.5298, 0.5245), (0.1333, 0.1476, 0.1590))
            ]
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

        #### Computing and distributing weights for weighted sampling ####
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
