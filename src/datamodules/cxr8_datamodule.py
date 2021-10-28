import torch
import os
import requests
from itertools import compress

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
        targets = np.array(targets)

        if np.sum(targets) == 0:
            targets = 6
        else:
            targets = np.argmax(targets)

        return img, targets

    def __len__(self):
        return len(self.targets)


class CXR8DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, resize, data_dir, num_workers, pin_memory, seed):
        super().__init__()
        self.name = "CXR8DataModule"

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
            csv_path=self.data_dir + "/CXR8/train_label_filtered.csv",
        )

        val = CXR8Dataset(
            img_path=self.data_dir + "/CXR8/images/",
            transform=transform_img,
            csv_path=self.data_dir + "/CXR8/val_label_filtered.csv",
        )

        self.test = CXR8Dataset(  # 12957
            img_path=self.data_dir + "/CXR8/images/",
            transform=transform_img,
            csv_path=self.data_dir + "/CXR8/test_label_filtered.csv",
        )

        self.ood_test = CXR8Dataset(  # 45824
            img_path=self.data_dir + "/CXR8/",
            transform=transform_img,
            csv_path=self.data_dir + "/CXR8/CheXpert-v1.0-small/train_filtered.csv",
        )

        self.train_enc, self.train_head = random_split(
            train,
            [50769, 1000],  # 51769
            generator=torch.Generator().manual_seed(self.seed),
        )

        self.val_enc, self.val_head = random_split(
            val,
            [6774, 700],  # 7474
            generator=torch.Generator().manual_seed(self.seed),
        )

        #### Computing and distributing weights for weighted sampling ####
        weights_train_head = torch.DoubleTensor(
            make_weights_for_balanced_classes(self.train_head, 7)
        )
        # weights_val_head = torch.DoubleTensor(
        #     make_weights_for_balanced_classes(self.val_head, 7)
        # )

        self.sampler_train_head = torch.utils.data.sampler.WeightedRandomSampler(
            weights_train_head, len(weights_train_head)
        )
        # self.sampler_val_head = torch.utils.data.sampler.WeightedRandomSampler(
        #     weights_val_head, len(weights_val_head)
        # )

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
            # sampler=self.sampler_val_head,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def ood_test_dataloader(self):
        return DataLoader(
            self.ood_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
