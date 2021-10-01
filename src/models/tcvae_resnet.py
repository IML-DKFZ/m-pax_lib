import math

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor

from torch.optim.lr_scheduler import CosineAnnealingLR

import torchvision
import pytorch_lightning as pl
from pl_bolts.models.autoencoders.basic_ae.basic_ae_module import AE

class betaTCVAE_ResNet(pl.LightningModule):
    def __init__(self,
                 trainset_size=50000,
                 latent_dim=10,
                 input_dim=32,
                 lr=0.001,
                 momentum=0.9,
                 weight_decay= 1e-4,
                 anneal_steps: int = 200,
                 alpha: float = 1.,
                 beta: float = 1.,
                 gamma: float = 1.
                 ):
        super(betaTCVAE_ResNet, self).__init__()
        self.save_hyperparameters()
        self.num_iter = 0

        # Encoder
        self.enc = torchvision.models.resnet50()
        self.enc.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.enc.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.enc_fc = nn.Linear(in_features=1000, out_features=256)
        self.fc_mu = nn.Linear(in_features=256, out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=256, out_features=latent_dim)

        # Decoder
        # From: https://github.com/AntixK/PyTorch-VAE/blob/master/models/betatc_vae.py

        hidden_dims = [32, 64, 128, 64, 32]
        modules = []

        if input_dim == 32:
            scale = 4
        else:
            scale = 256

        self.dec_fc = nn.Linear(latent_dim, scale *  8)

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.LeakyReLU())
            )
        
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1],
                                    hidden_dims[-1],
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    output_padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(hidden_dims[-1], out_channels= 1,
                            kernel_size= 3, padding= 1),
                nn.Sigmoid()
            )
        )

        self.dec = nn.Sequential(*modules)

    def encode(self, x):
        x = self.enc(x)

        x = F.relu(self.enc_fc(x))
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        return mu, log_var


    def sampling(self, mu, log_var):
        std = torch.exp(log_var * 0.5)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return z

    def decode(self, z):
        x = self.dec_fc(z)
        x = x.view(-1, 32, 8, 8)
        x = self.dec(x)
        return x

    def forward(self, x):
        mu, _ = self.encode(x)
        return mu

    def log_density_gaussian(self, x: Tensor, mu: Tensor, logvar: Tensor):
        norm = - 0.5 * (math.log(2 * math.pi) + logvar)
        log_density = norm - 0.5 * ((x - mu) ** 2 * torch.exp(-logvar))
        return log_density

    def loss(self, recons, x, mu, log_var, z):
        # Inspired by: https://github.com/YannDubs/disentangling-vae/blob/7b8285baa19d591cf34c652049884aca5d8acbca/disvae/models/losses.py#L316
        recons = torch.where(torch.isnan(recons), torch.zeros_like(recons), recons)
        recons = torch.where(torch.isinf(recons), torch.zeros_like(recons), recons)

        recons_loss = F.binary_cross_entropy(
            recons.view(-1, self.hparams.input_dim**2).clamp(0,1).type(torch.FloatTensor),
            x.view(-1, self.hparams.input_dim**2).clamp(0,1).type(torch.FloatTensor),
            reduction='sum')

        log_q_zx = self.log_density_gaussian(z, mu, log_var).sum(dim=1)

        zeros = torch.zeros_like(z)
        log_p_z = self.log_density_gaussian(z, zeros, zeros).sum(dim=1)

        batch_size, latent_dim = z.shape
        mat_log_q_z = self.log_density_gaussian(z.view(batch_size, 1, latent_dim),
                                                mu.view(1, batch_size,
                                                        latent_dim),
                                                log_var.view(1, batch_size, latent_dim))


        # Estimate the three KL terms (log(q(z))) via importance sampling
        strat_weight = (self.hparams.trainset_size - batch_size + 1) / \
            (self.hparams.trainset_size * (batch_size - 1))

        importance_weights = torch.Tensor(batch_size, batch_size).fill_(
            1 / (batch_size - 1)).to(x.device)

        importance_weights.view(-1)[::batch_size] = 1 / self.hparams.trainset_size
        importance_weights.view(-1)[1::batch_size] = strat_weight
        importance_weights[batch_size - 2, 0] = strat_weight
        log_importance_weights = importance_weights.log()

        mat_log_q_z += log_importance_weights.view(batch_size, batch_size, 1)

        log_q_z = torch.logsumexp(mat_log_q_z.sum(2), dim=1, keepdim=False)
        log_prod_q_z = torch.logsumexp(
            mat_log_q_z, dim=1, keepdim=False).sum(1)

        # Three KL Term components
        mi_loss = (log_q_zx - log_q_z).mean()
        tc_loss = (log_q_z - log_prod_q_z).mean()
        kld_loss = (log_prod_q_z - log_p_z).mean()

        if self.training:
            self.num_iter += 1
            anneal_rate = min(0 + 1 * self.num_iter /
                              self.hparams.anneal_steps, 1)
        else:
            anneal_rate = 1

        loss = recons_loss/batch_size + \
            self.hparams.alpha * mi_loss + \
            self.hparams.beta * tc_loss + \
            self.hparams.gamma * anneal_rate * kld_loss

        return loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
     
        mu, log_var = self.encode(x)
        z = self.sampling(mu, log_var)
     
        recons = self.decode(z)
   
        loss = self.loss(recons, x, mu, log_var, z)

        self.log('loss', loss, on_epoch=False, prog_bar=True, on_step=True, 
            sync_dist=True if torch.cuda.device_count() > 1 else False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch

        mu, log_var = self.encode(x)
        z = self.sampling(mu, log_var)

        recons = self.decode(z)


        val_loss = self.loss(recons, x, mu, log_var, z)

        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, 
            sync_dist=True if torch.cuda.device_count() > 1 else False)

        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay = self.hparams.weight_decay)

        scheduler = {'scheduler': CosineAnnealingLR(optimizer,T_max=self.trainer.max_epochs, verbose=True), 
                    'interval': 'epoch'}
        return [optimizer], [scheduler] 

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict
