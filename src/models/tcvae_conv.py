import torch
import torch.nn.functional as F
from torch import nn

from torch.optim.lr_scheduler import CosineAnnealingLR

import torchvision
import pytorch_lightning as pl

from src.utils.betatcvae_loss import betatc_loss
from src.utils.weights_init import weights_init


class betaTCVAE_Conv(pl.LightningModule):
    def __init__(
        self,
        trainset_size=50000,
        latent_dim=10,
        input_dim=32,
        input_channels=1,
        lr=0.001,
        momentum=0.9,
        weight_decay=1e-4,
        anneal_steps: int = 200,
        is_mss=True,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
    ):
        super(betaTCVAE_Conv, self).__init__()
        self.save_hyperparameters()
        self.num_iter = 0

        if input_dim == 32:
            hidden_dims = [32, 32, 64]
        else:
            hidden_dims = [32, 32, 32, 64, 64]

        modules = []

        modules.append(
            nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    out_channels=hidden_dims[1],
                    kernel_size=4,
                    padding=1,
                    stride=2,
                ),
                nn.ReLU(),
            )
        )

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.ReLU(),
                )
            )

        self.enc = nn.Sequential(*modules)

        self.enc_fc = nn.Linear(in_features=1024, out_features=256)
        self.fc_mu = nn.Linear(in_features=256, out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=256, out_features=latent_dim)

        # Decoder
        # From: https://github.com/AntixK/PyTorch-VAE/blob/master/models/betatc_vae.py

        hidden_dims = list(reversed(hidden_dims))
        modules = []

        self.dec_fc_1 = nn.Linear(latent_dim, 256)
        self.dec_fc_2 = nn.Linear(256, 1024)

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.ReLU(),
                )
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims[-1],
                    hidden_dims[-1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    hidden_dims[-1],
                    out_channels=input_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.Sigmoid(),
            )
        )

        self.dec = nn.Sequential(*modules)

        self.loss = betatc_loss(
            is_mss=is_mss,
            steps_anneal=anneal_steps,
            n_data=trainset_size,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

        self.init_weights()
        self.warm_up = 0

    def encoder(self, x):
        x = self.enc(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.enc_fc(x))
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        return mu, log_var

    def decoder(self, z):
        x = F.relu(self.dec_fc_1(z))
        x = F.relu(self.dec_fc_2(x))
        x = x.view(-1, 64, 4, 4)
        x = self.dec(x)
        return x

    def reparameterize(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.
        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)
        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size,
            latent_dim)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + std * eps
        else:
            # Reconstruction mode
            return mean

    def forward(self, x):
        """
        Forward pass of model.
        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        reconstruct = self.decoder(latent_sample)
        return reconstruct, latent_dist, latent_sample

    def init_weights(self):
        self.apply(weights_init)

    def sample_latent(self, x):
        """
        Returns a sample from the latent distribution.
        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        return latent_sample

    def training_step(self, batch, batch_idx):
        x, _ = batch

        recon_batch, latent_dist, latent_sample = self(x)

        rec_loss, kld = self.loss(
            x, recon_batch, latent_dist, self.training, latent_sample=latent_sample
        )

        if self.current_epoch == 0:
            self.warm_up += x.shape[0]
            warm_up_weight = self.warm_up / self.hparams.trainset_size
        else:
            warm_up_weight = 1

        loss = rec_loss + warm_up_weight * kld

        self.log(
            "kl_warm_up",
            warm_up_weight,
            on_epoch=False,
            prog_bar=True,
            on_step=True,
            sync_dist=True if torch.cuda.device_count() > 1 else False,
        )

        self.log(
            "kld",
            kld,
            on_epoch=False,
            prog_bar=True,
            on_step=True,
            sync_dist=True if torch.cuda.device_count() > 1 else False,
        )

        self.log(
            "rec_loss",
            rec_loss,
            on_epoch=False,
            prog_bar=True,
            on_step=True,
            sync_dist=True if torch.cuda.device_count() > 1 else False,
        )

        self.log(
            "loss",
            loss,
            on_epoch=False,
            prog_bar=True,
            on_step=True,
            sync_dist=True if torch.cuda.device_count() > 1 else False,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch

        recon_batch, latent_dist, latent_sample = self(x)

        rec_loss, kld_val = self.loss(
            x, recon_batch, latent_dist, self.training, latent_sample=latent_sample
        )

        val_loss = rec_loss + kld_val

        self.log(
            "val_loss",
            val_loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True if torch.cuda.device_count() > 1 else False,
        )

        self.log(
            "val_kld",
            kld_val,
            on_epoch=False,
            prog_bar=True,
            on_step=True,
            sync_dist=True if torch.cuda.device_count() > 1 else False,
        )

        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
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
