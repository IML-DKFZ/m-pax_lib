import torch
import os
import zipfile
import requests

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import pytorch_lightning as pl

# From: https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
# Computing class weights for balanced sampling 
def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def download_url(url, save_path): # Chunk wise downloading to not overuse RAM
    r = requests.get(url, stream=True, allow_redirects=True)
    with open(save_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024): 
            if chunk:
                f.write(chunk)
                f.flush()


class OCTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, resize, data_dir, num_workers, pin_memory, seed):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.resize = resize
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed

    def prepare_data(self):
        if not os.path.exists(os.path.join(self.data_dir, "OCT/test/")):
            data_url = "https://polybox.ethz.ch/index.php/s/kcRvwssD2yWGfsN/download"
            save_path = os.path.join(self.data_dir, "OCT/download_file.zip")

            print("Downloading and extracting OCT retina data...")

            download_url(data_url, save_path)

            zip_ref = zipfile.ZipFile(save_path, 'r')
            zip_ref.extractall(self.data_dir + "/OCT/")
            zip_ref.close()

            os.remove(save_path)

    def setup(self):
        transform_img = transforms.Compose([ 
            transforms.Resize((self.resize, self.resize)), # Bilinear resizing
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

        train = datasets.ImageFolder(self.data_dir + "/OCT/train", transform=transform_img)

        data_enc, self.train_head, self.val_head = random_split(train, [106000, 1700, 609], generator=torch.Generator().manual_seed(self.seed))

        self.train_enc, self.val_enc = random_split(data_enc, [84600, 21400], generator=torch.Generator().manual_seed(self.seed))

        self.test = datasets.ImageFolder(self.data_dir + "/OCT/test", transform=transform_img)

        #### Computing and distributing weights for weighted sampling ####
        weights = make_weights_for_balanced_classes(train.imgs, len(train.classes))
        weights = torch.DoubleTensor(weights)

        weights_enc, weights_head, _ = random_split(weights, [106000, 1700, 609], generator=torch.Generator().manual_seed(self.seed))

        weights_enc, _ = random_split(weights_enc, [84600, 21400], generator=torch.Generator().manual_seed(self.seed))

        self.sampler_head = torch.utils.data.sampler.WeightedRandomSampler(weights_head, len(weights_head))

        self.sampler_enc = torch.utils.data.sampler.WeightedRandomSampler(weights_enc, len(weights_enc))

    def train_dataloader(self):
        return DataLoader(self.train_enc, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory = self.pin_memory)

    def train_dataloader_head(self):
        return DataLoader(self.train_head, batch_size=self.batch_size, num_workers=self.num_workers, sampler=self.sampler_head, pin_memory = self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_enc, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory = self.pin_memory)

    def val_dataloader_head(self):
        return DataLoader(self.val_head, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory = self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory = self.pin_memory)
