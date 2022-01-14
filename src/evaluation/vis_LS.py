import imageio
import os

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision.utils import save_image

from src.evaluation.vis_helper import *

# To change name and type of the generated images
PLOT_NAMES = dict(
    generate_samples="samples.png",
    data_samples="data_samples.png",
    reconstruct="reconstruct.png",
    traversals="traversals.png",
    reconstruct_traverse="reconstruct_traverse.png",
    gif_traversals="posterior_traversals.gif",
)


class Visualizer:
    """Visualizer is used to generate images of samples, reconstructions,
    latent traversals of the trained model.
    """

    def __init__(self, index, latent_dim, max_traversal, model, model_dir):
        """Called upon initialization.

        Parameters
        ----------
        index : int
            Index of the image from whereon the next ten samples are visualized.
        latent_dim : int
            Latent dimension of the encoder.
        max_traversal : float
            The maximum displacement induced by a latent traversal. Symmetrical
            traversals are assumed. Note in the case of the posterior, the distribution
            is not standard normal anymore.
        model : src.models.tcvae_conv.betaTCVAE_Conv or src.models.tcvae_resnet.betaTCVAE_ResNet
            The beta-TCVAE model with encoder and decoder.
        model_dir : str
            The directory that the model is saved to.
        """
        self.device = next(self.model.parameters()).device
        self.index = index
        self.latent_dim = latent_dim
        self.max_traversal = max_traversal
        self.model = model
        self.model_dir = model_dir

    def _get_traversal_range(self, mean=0, std=1):
        """Returns the corresponding traversal range.

        Parameters
        ----------
        mean : int, optional
            Mean of the normal distribution, by default 0.
        std : int, optional
            Standard deviation of the normal distribution, by default 1.

        Returns
        -------
        dict
            Symmetric traversal range.
        """
        # symmetrical traversals
        return (mean - std * self.max_traversal, mean + std * self.max_traversal)

    def _traverse_line(self, idx, n_samples, data=None):
        """Return a (size, latent_size) latent sample, corresponding to a traversal
        of a latent variable indicated by idx.

        Parameters
        ----------
        idx : int
            Index of continuous dimension to traverse. If the continuous latent
            vector is 10 dimensional and idx = 7, then the 7th dimension
            will be traversed while all others are fixed.
        n_samples : int
            Number of samples to generate.
        data : torch.Tensor or None, optional
            Data to use for computing the posterior. Shape (N, C, H, W). If
            `None` then use the mean of the prior (all zeros) for all other dimensions.

        Returns
        -------
        torch.Tensor
            Traversal for a latent sample.

        Raises
        ------
        ValueError
            If more than one obervation is submitted.
        """
        if data is None:
            # mean of prior for other dimensions
            samples = torch.zeros(n_samples, self.latent_dim)
            traversals = torch.linspace(*self._get_traversal_range(), steps=n_samples)

        else:
            if data.size(0) > 1:
                raise ValueError(
                    "Every value should be sampled from the same posterior, but {} datapoints given.".format(
                        data.size(0)
                    )
                )

            with torch.no_grad():
                post_mean, post_logvar = self.model.encoder(data.to(self.device))
                samples = self.model.reparameterize(post_mean, post_logvar)
                samples = samples.cpu().repeat(n_samples, 1)
                post_mean_idx = post_mean.cpu()[0, idx]
                post_std_idx = torch.exp(post_logvar / 2).cpu()[0, idx]

            # travers from the gaussian of the posterior in case quantile
            traversals = torch.linspace(
                *self._get_traversal_range(mean=post_mean_idx, std=post_std_idx),
                steps=n_samples
            )

        for i in range(n_samples):
            samples[i, idx] = traversals[i]

        return samples

    def save_plot(self, to_plot, size, filename):
        """Create plot and save or return it.

        Parameters
        ----------
        to_plot : torch.Tensor
            Image to plot.
        size : tuple of ints
            Dimensions of image.
        filename : str
            Name and type of image file.

        Raises
        ------
        ValueError
            Wrong image dimensions.
        """
        to_plot = F.interpolate(to_plot)

        if size[0] * size[1] != to_plot.shape[0]:
            raise ValueError(
                "Wrong size {} for datashape {}".format(size, to_plot.shape)
            )

        # `nrow` is number of images PER row => number of col
        kwargs = dict(nrow=size[1], pad_value=0)
        filename = os.path.join(self.model_dir, filename)
        save_image(to_plot, filename, **kwargs)

    def _decode_latents(self, latent_samples):
        """Decodes latent samples into images.
        Parameters
        ----------
        latent_samples : torch.Tensor
            Samples from latent distribution. Shape (N, L) where L is dimension
            of latent distribution.

        Returns
        -------
        torch.Tensor
            Reconstructed image from decoder.
        """
        latent_samples = latent_samples.to(self.device)
        return self.model.decoder(latent_samples).cpu()

    def generate_samples(self, size=(8, 8)):
        """Plot generated samples from the prior and decoding.

        Parameters
        ----------
        size : tuple of ints, optional
            Size of the final grid.
        """
        prior_samples = torch.randn(size[0] * size[1], self.latent_dim)
        generated = self._decode_latents(prior_samples)
        return self.save_plot(generated.data, size, PLOT_NAMES["generate_samples"])

    def data_samples(self, data, size=(8, 8)):
        """Plot samples from the data
        Parameters
        ----------
        data : torch.Tensor
            Data to be reconstructed. Shape (N, C, H, W)
        size : tuple of ints, optional
            Size of the final grid.
        """
        data = data[: size[0] * size[1], ...]
        return self.save_plot(data, size, PLOT_NAMES["data_samples"])

    def reconstruct(self, data, size=(8, 8), is_original=True, is_force_return=False):
        """Generate reconstructions of data through the model.

        Parameters
        ----------
        data : torch.Tensor
            Data to be reconstructed. Shape (N, C, H, W)
        size : tuple of ints, optional
            Size of grid on which reconstructions will be plotted. The number
            of rows should be even when `is_original`, so that upper
            half contains true data and bottom half contains reconstructions.contains
        is_original : bool, optional
            Whether to exclude the original plots.
        is_force_return : bool, optional
            Force returning instead of saving the image.

        Raises
        ------
        ValueError
            When original images should also be show, rows have to be even.
        """
        if is_original:
            if size[0] % 2 != 0:
                raise ValueError(
                    "Should be even number of rows when showing originals not {}".format(
                        size[0]
                    )
                )
            n_samples = size[0] // 2 * size[1]
        else:
            n_samples = size[0] * size[1]

        with torch.no_grad():
            originals = data.to(self.device)[:n_samples, ...]
            recs, _, _ = self.model(originals)

        originals = originals.cpu()
        recs = recs.cpu()

        to_plot = torch.cat([originals, recs]) if is_original else recs
        return self.save_plot(
            to_plot, size, PLOT_NAMES["reconstruct"], is_force_return=is_force_return
        )

    def traversals(
        self, data=None, n_per_latent=8, n_latents=None, is_force_return=False
    ):
        """Plot traverse through all latent dimensions (prior or posterior) one
        by one and plots a grid of images where each row corresponds to a latent
        traversal of one latent dimension.

        Parameters
        ----------
        data : bool, optional
            Data to use for computing the latent posterior. If `None` traverses
            the prior.
        n_per_latent : int, optional
            The number of points to include in the traversal of a latent dimension.
            I.e. number of columns.
        n_latents : int, optional
            The number of latent dimensions to display. I.e. number of rows. If `None`
            uses all latents.
        is_force_return : bool, optional
            Force returning instead of saving the image.
        """
        n_latents = n_latents if n_latents is not None else self.latent_dim
        latent_samples = [
            self._traverse_line(dim, n_per_latent, data=data)
            for dim in range(self.latent_dim)
        ]
        # To change order of latent dimensions:
        # latent_samples = latent_samples.iloc[[1,2,7,0,5,8,6,3,9,4],:]
        decoded_traversal = self._decode_latents(torch.cat(latent_samples, dim=0))

        decoded_traversal = decoded_traversal[range(n_per_latent * n_latents), ...]

        size = (n_latents, n_per_latent)
        sampling_type = "prior" if data is None else "posterior"
        filename = "{}_{}".format(sampling_type, PLOT_NAMES["traversals"])

        return self.save_plot(
            decoded_traversal.data, size, filename, is_force_return=is_force_return
        )

    def reconstruct_traverse(
        self, dataloader, is_posterior=True, n_per_latent=8, n_latents=None
    ):
        """Creates a figure whith first row for original images, second are
        reconstructions, rest are traversals (prior or posterior) of the latent
        dimensions.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            Dataloader for data to be reconstructed.
        is_posterior : bool, optional
            Whether to sample from the posterior.
        n_per_latent : int, optional
            The number of points to include in the traversal of a latent dimension.
            I.e. number of columns.
        n_latents : int, optional
            The number of latent dimensions to display. I.e. number of rows. If `None`
            uses all latents.
        """
        data, _ = next(iter(dataloader))

        n_latents = n_latents if n_latents is not None else self.latent_dim

        reconstructions = self.reconstruct(
            data[: 2 * n_per_latent, ...], size=(2, n_per_latent), is_force_return=True
        )
        traversals = self.traversals(
            data=data[self.index].unsqueeze(0) if is_posterior else None,
            n_per_latent=n_per_latent,
            n_latents=n_latents,
            is_force_return=True,
        )

        concatenated = np.concatenate((reconstructions, traversals), axis=0)
        concatenated = Image.fromarray(concatenated)

        filename = os.path.join(self.model_dir, PLOT_NAMES["reconstruct_traverse"])
        concatenated.save(filename)

    def gif_traversals(self, dataloader, n_latents=None, n_per_gif=15, index=0):
        """Generates a grid of gifs of latent posterior traversals where the rows
        are the latent dimensions and the columns are random images.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            Dataloader for data to use for computing the latent posteriors. The number of datapoint
            (batchsize) will determine the number of columns of the grid.
        n_latents : int, optional
            The number of latent dimensions to display. I.e. number of rows. If `None`
            uses all latents.
        n_per_gif : int, optional
            Number of images per gif (number of traversals)
        index : int, optional
            Index of the image from whereon the next gifs are visualized, by default 0.
        """
        data, _ = next(iter(dataloader))
        data = data[index : (index + 5)]
        n_images, _, _, width_col = data.shape
        width_col = int(width_col)
        all_cols = [[] for c in range(n_per_gif)]
        for i in range(n_images):
            grid = self.traversals(
                data=data[i : i + 1, ...],
                n_per_latent=n_per_gif,
                n_latents=n_latents,
                is_force_return=True,
            )

            height, width, c = grid.shape
            padding_width = (width - width_col * n_per_gif) // (n_per_gif + 1)

            # split the grids into a list of column images (and removes padding)
            for j in range(n_per_gif):
                all_cols[j].append(
                    grid[
                        :,
                        [
                            (j + 1) * padding_width + j * width_col + i
                            for i in range(width_col)
                        ],
                        :,
                    ]
                )

        all_cols = [
            concatenate_pad(cols, pad_size=7, pad_values=255, axis=1)
            for cols in all_cols
        ]

        filename = os.path.join(self.model_dir, PLOT_NAMES["gif_traversals"])
        imageio.mimsave(filename, all_cols, fps=10)
