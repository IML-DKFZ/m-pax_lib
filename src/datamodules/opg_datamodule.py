import os
import numpy as np
from PIL import Image
import torch
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader, Subset
from scipy.ndimage.measurements import center_of_mass
import pandas as pd
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import pytorch_lightning as pl
from torchvision import transforms as transform_lib


def extract_ROI(image, mask, window_size):

    # Get mask centroid
    c1, c2 = center_of_mass(mask)
    c1, c2 = int(c1), int(c2)
    # Reshape image according to window_size
    image = image[
        c1 - int(window_size / 2) : c1 + int(window_size / 2),
        c2 - int(window_size / 2) : c2 + int(window_size / 2),
    ]

    return image


def normalization():
    mean = 12526.53
    std = 30368.829025877254

    max_val = 65535.0
    normalize = transform_lib.Normalize(mean=[mean / max_val], std=[std / max_val])
    return normalize


class DentalImagingDataset(Dataset):
    def __init__(
        self,
        images_dir,
        csv_file,
        molar_guarantee=True,
        transform=None,
        considered_class=1,  # considered_class 1,2 or 3 (see annotations.csv)
        ROI_size=256,
    ):

        # Choose labels from annotations

        self.annotations = pd.read_csv(csv_file)
        self.considered_class = considered_class
        if considered_class == 1:
            self.pad = 1
        elif considered_class == 2:
            self.pad = 4
        else:
            self.pad = 7
        if molar_guarantee is True:  #
            self.annotations = self.annotations[self.annotations["molar_yn"] == 1].iloc[
                :, [0, 2, 3, 4]
            ]
            self.considered_class -= 1

        self.annotations = self.annotations[
            (self.annotations.File_name != "20_02.dcm-71_l")
            & (self.annotations.File_name != "19_11.dcm-20_l")
        ]
        self.annotations.dropna(inplace=True)

        self.annotations.reset_index(drop=True, inplace=True)

        self.imgs_dir = images_dir
        self.transform = transform
        self.considered_class = considered_class
        self.ROI_size = ROI_size

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, item):
        # Image
        file_name = self.annotations.iloc[item, 0] + ".dcm"
        mask_name = file_name[:-4] + ".gipl"
        file_path = os.path.join(self.imgs_dir, file_name)
        mask_path = os.path.join(self.imgs_dir, mask_name)
        image = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
        image = image.astype(np.float32).reshape(image.shape[1], image.shape[2])
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        mask = mask.astype(np.float32).reshape(mask.shape[1], mask.shape[2])

        # if self.ROI_size is not None:
        ############### TRAIN WITH ROIS ##################
        # Basic transforms: ROI extraction, to tensor, normalize
        image = extract_ROI(image, mask, self.ROI_size).astype(np.int64)
        ROI_width = 2 * self.ROI_size
        image = np.pad(
            image,
            ((0, ROI_width - image.shape[0]), (ROI_width - image.shape[1], 0)),
            "constant",
            constant_values=image.min(),
        )

        # image = image.astype(np.float32)
        image = Image.fromarray(image.astype(np.uint8), mode="L")
        if self.transform is not None:
            image = self.transform(image)

        # Labels
        labels = int(self.annotations.iloc[item, self.considered_class] - self.pad)

        return image, labels


class DentalImagingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        resize,
        molar_guarantee,
        considered_class,
        data_dir,
        labelled_size,
        num_workers,
        pin_memory,
        seed,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.image_dir = os.path.join(self.data_dir, "images_nmasks")
        self.csv_file = os.path.join(data_dir, "annotations_from_cv.csv")
        # self.csv_file = os.path.join(data_dir,'annotations_final.csv')
        self.molar_guarantee = molar_guarantee
        self.considered_class = considered_class
        self.batch_size = batch_size
        self.resize = resize
        self.labelled_size = labelled_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed

    def prepare_data(self):
        pass

    def setup(self):
        transform_img = transforms.Compose(
            [
                transforms.Resize((self.resize, self.resize)),
                transforms.ToTensor(),
                normalization(),
            ]
        )

        dataset = DentalImagingDataset(
            images_dir=self.image_dir,
            csv_file=self.csv_file,
            molar_guarantee=self.molar_guarantee,
            considered_class=self.considered_class,
            ROI_size=self.resize,
            transform=transform_img,
        )

        data_size = len(dataset)
        test_size = int(0.2 * data_size)
        non_test_size = data_size - test_size
        labelled_size = int(non_test_size * self.labelled_size)

        # Head
        val_head_size = int(labelled_size * 0.2)
        train_head_size = labelled_size - val_head_size

        # Encoder
        val_enc_size = int((non_test_size - labelled_size) * 0.2)
        train_enc_size = (non_test_size - labelled_size) - val_enc_size

        # train-test split
        idxs = list(range(data_size))
        non_test_ds = Subset(dataset, idxs[test_size:])
        self.test = Subset(dataset, idxs[:test_size])

        (self.train_enc, self.val_enc, self.train_head, self.val_head) = random_split(
            non_test_ds,
            [train_enc_size, val_enc_size, train_head_size, val_head_size],
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
