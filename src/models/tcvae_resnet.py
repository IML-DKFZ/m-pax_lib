import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision

from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils.betatcvae_loss import betatc_loss
from src.utils.weights_init import weights_init


class betaTCVAE_ResNet(pl.LightningModule):
    """beta-TCVAE model with encoder and decoder, inherit from Lightning module.
    Contains the following methods:
     - encoder
     - decoder
     - reparameterize
     - forward
     - init_weights
     - sample_latent
     - training_step
     - validation_step
     - configure_optimizers
     - get_progress_bar_dict
    """

    def __init__(
        self,
        alpha: float = 1.0,
        anneal_steps: int = 200,
        beta: float = 1.0,
        gamma: float = 1.0,
        input_channels=1,
        input_dim=32,
        is_mss=True,
        latent_dim=10,
        lr=0.001,
        momentum=0.9,
        trainset_size=50000,
        weight_decay=1e-4,
    ):
        """Called upon initialization. Constructs resnet encoder and decoder architecture,
        initializes weights and loss.

        Parameters
        ----------
        alpha : float, optional
            Weight of the mutual information term, by default 1.0.
        anneal_steps : int, optional
            Number of annealing steps where gradually adding the regularisation, by default 200.
        beta : float, optional
            Weight of the total correlation term, by default 1.0.
        gamma : float, optional
            Weight of the dimension-wise KL term, by default 1.0.
        input_channels : int, optional
            Input channels for colored or black and white image, by default 1.
        input_dim : int, optional
            Dimension of quadratic input image, by default 32.
        is_mss : bool, optional
            Whether to use minibatch stratified sampling instead of minibatch
            weighted sampling.
        latent_dim : int, optional
            Latent dimension of encoder, by default 10.
        lr : float, optional
            Learning rate value for optimizer, by default 0.001.
        momentum : float, optional
            Momentum value for optimizer, by default 0.9.
        trainset_size : int, optional
            Size of the training dataset, by default 50000.
        weight_decay : float, optional
            Weight decay regularization value to prevent overfitting, by default 1e-4.
        """
        super(betaTCVAE_ResNet, self).__init__()
        self.save_hyperparameters()
        self.num_iter = 0

        # Encoder
        if input_dim == 32:
            self.enc = torchvision.models.resnet18()
            hidden_dims = [32, 64, 32]
        elif input_dim == 128:
            self.enc = torchvision.models.resnet18()
            hidden_dims = [32, 256, 128, 64, 32]
        else:
            self.enc = torchvision.models.resnet50()
            hidden_dims = [32, 512, 256, 128, 64, 32]

        self.enc.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=1, padding=3, bias=False
        )
        self.enc.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.enc_fc = nn.Linear(in_features=1000, out_features=256)
        self.fc_mu = nn.Linear(in_features=256, out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=256, out_features=latent_dim)

        # Decoder
        modules = []

        self.dec_fc = nn.Linear(latent_dim, 512)

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
                    nn.LeakyReLU(),
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
                nn.LeakyReLU(),
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
        """Takes image(s), returns mean and log variance vectors of latent space.

        Parameters
        ----------
        x : torch.Tensor
            Input image(s).

        Returns
        -------
        torch.Tensor, torch.Tensor
            Mean and log variance.
        """
        x = self.enc(x)

        x = F.relu(self.enc_fc(x))
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        return mu, log_var

    def decoder(self, z):
        """Reconstructs image(s) from latent samples

        Parameters
        ----------
        z : torch.Tensor
            Latent samples.

        Returns
        -------
        torch.Tensor
            Reconstructed image(s).
        """
        x = self.dec_fc(z)
        x = x.view(-1, 32, 4, 4)
        x = self.dec(x)
        return x

    def reparameterize(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim).
        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size,
            latent_dim).

        Returns
        -------
        torch.Tensor
            Returns sampled value or maximum a posteriori (MAP) estimator.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + std * eps
        else:
            # Reconstruction mode
            return mean

    def forward(self, x):
        """Forward pass of model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)

        Returns
        -------
        torch.Tensor, torch.Tensor, torch.Tensor
            Returns reconstructed image(s), latent distributions (mean and log variance), and latent samples.
        """
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        reconstruct = self.decoder(latent_sample)
        return reconstruct, latent_dist, latent_sample

    def init_weights(self):
        self.apply(weights_init)

    def sample_latent(self, x):
        """Returns a sample from the latent distribution.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)

        Returns
        -------
        torch.Tensor
            Samples from latent distribution.
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
        """Optimizer and learning rate scheduler configuration.

        Returns
        -------
        torch.optim.Adam, torch.optim.lr_scheduler.CosineAnnealingLR
            Returns optimizer and learning rate scheduler.
        """
        optimizer = Adam(
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
        """Remove "v_num" from progress bar.

        Returns
        -------
        dict
            Dictionary with all tqdm relevant information.
        """
        tqdm_dict = super().get_progress_bar_dict()
        if "v_num" in tqdm_dict:
            del tqdm_dict["v_num"]
        return tqdm_dict
