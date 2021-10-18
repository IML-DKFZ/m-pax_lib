import random

import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
import torch
import imageio

from torchvision.utils import make_grid


def get_samples(dataloader, num_samples, idcs=[]):
    """Generate a number of samples from the dataset.
    Parameters
    ----------
    dataset : str
        The name of the dataset.
    num_samples : int, optional
        The number of samples to load from the dataset
    idcs : list of ints, optional
        List of indices to of images to put at the begning of the samples.
    """
    data_loader = dataloader.train_dataloader().dataset

    idcs += random.sample(range(len(data_loader)), num_samples - len(idcs))
    samples = torch.stack([data_loader[i][0] for i in idcs], dim=0)

    return samples


def add_labels(input_image, labels):
    """Adds labels next to rows of an image.
    Parameters
    ----------
    input_image : image
        The image to which to add the labels
    labels : list
        The list of labels to plot
    """
    new_width = input_image.width + 100
    new_size = (new_width, input_image.height)
    new_img = Image.new("RGB", new_size, color="white")
    new_img.paste(input_image, (0, 0))
    draw = ImageDraw.Draw(new_img)

    for i, s in enumerate(labels):
        draw.text(
            xy=(
                new_width - 100 + 0.005,
                int((i / len(labels) + 1 / (2 * len(labels))) * input_image.height),
            ),
            text=s,
            fill=(0, 0, 0),
        )

    return new_img


def make_grid_img(tensor, **kwargs):
    """Converts a tensor to a grid of images that can be read by imageio.
    Notes
    -----
    * from in https://github.com/pytorch/vision/blob/master/torchvision/utils.py
    Parameters
    ----------
    tensor (torch.Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
        or a list of images all of the same size.
    kwargs:
        Additional arguments to `make_grid_img`.
    """
    grid = make_grid(tensor, **kwargs)
    img_grid = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    img_grid = img_grid.to("cpu", torch.uint8).numpy()
    return img_grid


def get_image_list(image_file_name_list):
    image_list = []
    for file_name in image_file_name_list:
        image_list.append(Image.open(file_name))
    return image_list


def arr_im_convert(arr, convert="RGBA"):
    """Convert an image array."""
    return np.asarray(Image.fromarray(arr).convert(convert))


def plot_grid_gifs(filename, grid_files, pad_size=7, pad_values=255):
    """Take a grid of gif files and merge them in order with padding."""
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
    """Concatenate lsit of array with padding inbetween."""
    pad = np.ones_like(arrays[0]).take(indices=range(pad_size), axis=axis) * pad_values

    new_arrays = [pad]
    for arr in arrays:
        new_arrays += [arr, pad]
    new_arrays += [pad]
    return np.concatenate(new_arrays, axis=axis)
