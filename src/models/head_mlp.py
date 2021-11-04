import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD, Adam
from torchmetrics.functional import confusion_matrix, accuracy

import pytorch_lightning as pl

from src.models.tcvae_resnet import *
from src.models.tcvae_conv import *


class MLP(pl.LightningModule):
    def __init__(
        self,
        folder_name: str,
        data_dir: str,
        latent_dim: int,
        num_classes: int,
        weight_decay: float,
        momentum=0.9999,
        lr=0.0001,
        blocked_latent_features=[],
        fix_weights=True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.kept_latent_features = torch.tensor(
            [x for x in list(range(0, latent_dim)) if x not in blocked_latent_features]
        )

        path_ckpt = data_dir + "/models/" + folder_name + "/encoder.ckpt"

        for architecture in [betaTCVAE_ResNet, betaTCVAE_Conv]:
            try:
                self.encoder = architecture.load_from_checkpoint(path_ckpt)
                break
            except RuntimeError:
                # repeat the loop on failure
                continue

        if fix_weights == True:
            self.encoder.freeze()
        else:
            self.encoder.eval()

        self.fc1 = nn.Linear(len(self.kept_latent_features), 512, bias=True)
        self.fc2 = nn.Linear(512, 512, bias=True)
        self.fc3 = nn.Linear(512, self.hparams.num_classes, bias=True)

    def forward(self, x):

        if x.shape[1] != self.hparams.latent_dim:
            x, _ = self.encoder.encoder(x)

        if len(self.hparams.blocked_latent_features) > 0:
            x = x.index_select(1, self.kept_latent_features.to(x.device))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y_hat = F.softmax(self.fc3(x), dim=1)

        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y, reduction="mean")

        acc = accuracy(
            y_hat,
            y,
            average="macro",
            num_classes=self.hparams.num_classes,
        )

        self.log(
            "acc",
            acc,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True if torch.cuda.device_count() > 1 else False,
        )

        self.log(
            "loss",
            loss,
            on_epoch=True,
            sync_dist=True if torch.cuda.device_count() > 1 else False,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        val_loss = F.cross_entropy(y_hat, y, reduction="mean")

        return {"val_loss": val_loss, "val_y": y, "val_y_hat": y_hat}

    def validation_epoch_end(self, outputs):
        val_y = torch.cat(tuple([x["val_y"] for x in outputs]))
        val_y_hat = torch.cat(tuple([x["val_y_hat"] for x in outputs]))

        val_loss = F.cross_entropy(val_y_hat, val_y, reduction="mean")

        acc = accuracy(
            val_y_hat,
            val_y,
            average="macro",
            num_classes=self.hparams.num_classes,
        )

        self.log(
            "val_acc",
            acc,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True if torch.cuda.device_count() > 1 else False,
        )

        self.log(
            "val_loss",
            val_loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True if torch.cuda.device_count() > 1 else False,
        )

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        test_loss = F.cross_entropy(y_hat, y, reduction="mean")

        return {"test_loss": test_loss, "test_y": y, "test_y_hat": y_hat}

    def test_epoch_end(self, outputs):
        test_y = torch.cat(tuple([x["test_y"] for x in outputs]))
        test_y_hat = torch.cat(tuple([x["test_y_hat"] for x in outputs]))

        test_loss = F.cross_entropy(test_y_hat, test_y, reduction="mean")

        acc = accuracy(
            test_y_hat,
            test_y,
            average="macro",
            num_classes=self.hparams.num_classes,
        )
        confmat = confusion_matrix(
            test_y_hat, test_y, num_classes=self.hparams.num_classes
        )

        self.log(
            "test_loss",
            test_loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True if torch.cuda.device_count() > 1 else False,
        )

        self.log(
            "test_acc",
            acc,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True if torch.cuda.device_count() > 1 else False,
        )

        print(
            "\n Confusion Matrix: \n",
            torch.round(confmat.type(torch.FloatTensor)).type(torch.IntTensor),
        )

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.lr
        )

        scheduler = {
            "scheduler": CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs, verbose=True
            ),
            "interval": "epoch",
        }
        return [optimizer], [scheduler]

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if "v_num" in tqdm_dict:
            del tqdm_dict["v_num"]
        return tqdm_dict
