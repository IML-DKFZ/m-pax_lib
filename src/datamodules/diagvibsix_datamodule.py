import pickle

import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
import pytorch_lightning as pl


class DiagVibSixDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        x = self.images[index]

        y = self.labels[index]

        if self.transform:
            x = self.transform(x)
            x = x.permute(1, 0, 2)

        return x, y


class DiagVibSixDataModule(pl.LightningDataModule):
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
        transform_img = transforms.Compose([transforms.ToTensor()])

        file = open(self.data_dir + "/DiagVibSix/train.pkl", "rb")
        train_data = pickle.load(file)
        file.close()

        self.train = DiagVibSixDataset(
            train_data["images"], train_data["task_labels"], transform=transform_img
        )

        file = open(self.data_dir + "/DiagVibSix/val.pkl", "rb")
        val_data = pickle.load(file)
        file.close()

        self.val = DiagVibSixDataset(
            val_data["images"], val_data["task_labels"], transform=transform_img
        )

        file = open(self.data_dir + "/DiagVibSix/test.pkl", "rb")
        test_data = pickle.load(file)
        file.close()

        self.test = DiagVibSixDataset(
            test_data["images"], test_data["task_labels"], transform=transform_img
        )

        self.train_enc, self.train_head = random_split(  # 43740
            self.train, [43000, 740], generator=torch.Generator().manual_seed(self.seed)
        )

        self.val_enc, self.val_head = random_split(  # 8748
            self.val, [8100, 648], generator=torch.Generator().manual_seed(self.seed)
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
