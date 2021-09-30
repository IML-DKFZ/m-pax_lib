import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.utils

"""
Creates an object to visualize the original images and 
reconstructions produced by the decoder. Also the encoding 
of the image which is used to visualize the LSFs is selected.
"""

class vis_Reconstructions:
    def __init__(self, model, datamodule, type):
        self.model = model
        self.model.eval()

        self.datamodule = datamodule
        self.type = type

    def to_img(self, x):
        x = x.clamp(0, 1)
        return x

    def show_image(self, img):
        img = self.to_img(img)
        npimg = img.numpy()
        plt.figure(figsize=(12, 6), dpi=200)
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis('off')

    def visualise_output(self, images):
        with torch.no_grad():
            
            height = images[0].shape[1]
            width = images[0].shape[2]

            model = self.model

            if self.type=='betaVAE' or self.type=='betaTCVAE':
                mu, log_var = model.encode(images.view(-1, height * width))
            else:
                mu, log_var = model.encode(images)

            # Select image for LSF visualization:
            self.mu = mu[21] # Can be changed, 21 gives a nice result for the default seed
            self.sd = torch.exp(log_var[21] * 0.5)

            z = model.sampling(mu, log_var)
            images = model.decode(z)
            images = images.cpu()

            if self.type == 'betaVAE' or self.type == 'betaTCVAE':
                images = self.to_img(images.view(-1, 1, height, width))
            else:
                images = self.to_img(images)
            
            np_imagegrid = torchvision.utils.make_grid(images[0:50], 10, 5).numpy()
            plt.figure(figsize=(12, 6), dpi=200)
            plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
            plt.axis('off')


    def visualise(self):
        images, labels = iter(self.datamodule).next()

        print('\nVisualizing Original Images...')
        self.show_image(torchvision.utils.make_grid(images[0:50], 10, 5))
        plt.savefig('./images/original.png')

        print('Visualizing Reconstructions...')
        self.visualise_output(images)
        plt.savefig('./images/reconstructions.png')
        return self.mu, self.sd