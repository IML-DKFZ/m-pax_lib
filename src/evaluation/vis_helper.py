import imageio
import random

import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
import torch

from torchvision.utils import make_grid


def make_grid_img(tensor, **kwargs):
    """Converts a tensor to a grid of images that can be read by imageio.

    Parameters
    ----------
    tensor : torch.Tensor or list
        4D mini-batch Tensor of shape (B x C x H x W) or a list of images all of the same size.

    Returns
    -------
    numpy.array
        Numpy array grid which can be processed by imageio.
    """
    grid = make_grid(tensor, **kwargs)
    img_grid = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    img_grid = img_grid.to("cpu", torch.uint8).numpy()
    return img_grid


def arr_im_convert(arr, convert="RGBA"):
    """Convert an image array. Used to output gif.

    Parameters
    ----------
    arr : numpy.array
        Image to be converted.
    convert : str, optional
        The three-channel (RGB) color model to be output, by default "RGBA".

    Returns
    -------
    numpy.array
        Converted image.
    """
    return np.asarray(Image.fromarray(arr).convert(convert))


def plot_grid_gifs(filename, grid_files, pad_size=7, pad_values=255):
    """Take a grid of gif files and merge them in order with padding.

    Parameters
    ----------
    filename : str
        Name with type ending of image file.
    grid_files : list
        Grid with images for every gif.
    pad_size : int, optional
        Padding between gifs, by default 7.
    pad_values : int, optional
        Padding color, by default 255.
    """
    grid_gifs = [[imageio.mimread(f) for f in row] for row in grid_files]
    n_per_gif = len(grid_gifs[0][0])

    # convert all to RGBA which is the most general => can merge any image
    imgs = [
        concatenate_pad(
            [
                concatenate_pad(
                    [arr_im_convert(gif[i], convert="RGBA") for gif in row],
                    pad_size,
                    pad_values,
                    axis=1,
                )
                for row in grid_gifs
            ],
            pad_size,
            pad_values,
            axis=0,
        )
        for i in range(n_per_gif)
    ]

    imageio.mimsave(filename, imgs, fps=12)


def concatenate_pad(arrays, pad_size, pad_values, axis=0):
    """Concatenate list of array with padding inbetween.

    Parameters
    ----------
    arrays : list
        List with all gif arrays.
    pad_size : int
        Padding between gifs.
    pad_values : int
        Padding color.
    axis : int, optional
        Axe alignment of the gifs, by default 0

    Returns
    -------
    np.array
        New image array with padding between gifs.
    """
    pad = np.ones_like(arrays[0]).take(indices=range(pad_size), axis=axis) * pad_values

    new_arrays = [pad]
    for arr in arrays:
        new_arrays += [arr, pad]
    new_arrays += [pad]
    return np.concatenate(new_arrays, axis=axis)
