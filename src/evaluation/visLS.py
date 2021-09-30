import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.utils import make_grid

"""
Creates an object to sample and visualize the effect of the LSFs
by sampling from the conditional latent distributions. 
"""

class vis_LatentSpace:
    def __init__(self, model, mu, sd, latent_dim=10, latent_range=3, input_dim=28):
        self.model = model
        self.model.eval()

        self.latent_dim = latent_dim
        self.latent_range = latent_range
        self.input_dim = input_dim
        self.mu = mu
        self.sd = sd

    def to_img(self, x):
        x = x.clamp(0, 1)
        return x

    def show_image(self, img):
        img = self.to_img(img)
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    def visualise(self):
        print("Visualizing LSFs...")
        recon = []

        for i in range(0,self.latent_dim,1):
            latent = self.mu
            latent = torch.transpose(latent.repeat(20, 1), 0, 1)

            latent[i, :] = torch.linspace(
                self.latent_range*-self.sd[i], 
                self.latent_range*self.sd[i], 20) + self.mu[i]

            latent = torch.transpose(latent, 0, 1)
            img_recon = self.model.decode(latent)
            recon.append(img_recon.view(-1, 1, self.input_dim, self.input_dim))

        recon = torch.cat(recon)

        fig, ax = plt.subplots()
        plt.figure(figsize=(15, 15), dpi=200)
        plt.axis('off')
        self.show_image(make_grid(recon.data, 20, 8))

        if self.input_dim == 28:
            step_size = 36
        elif self.input_dim == 64:
            step_size = 74
        else:
            step_size = 340

        for i in range(0, self.latent_dim, 1):
            plt.text(5, (self.input_dim/2.1) + (i*step_size), str(i), color="red", fontsize = 14)

        plt.savefig('./images/latent space features.png')