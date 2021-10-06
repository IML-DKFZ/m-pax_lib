import os
from typing import List, Optional

import hydra
from omegaconf import DictConfig

import pytorch_lightning as pl
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    seed_everything,
)
import torch

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from src.models.tcvae_resnet import betaTCVAE_ResNet

from src.evaluation.vis_LS import *
from src.evaluation.vis_AM import *

from src.utils import utils
log = utils.get_logger(__name__)

def evaluate(config: DictConfig) -> Optional[float]:
    """Main function for plotting fro pretrained models.
    Parameters
    ----------
    config: config Arguments
    """

    # Set Seed
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule.setup()

    # Init model and paths
    log.info("Instantiating model")

    output_dir = config.data_dir + "models/" + config.evaluation.model_dir + "/images/"

    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)

    path_ckpt = config.data_dir + "models/" + config.evaluation.model_dir + "/encoder.ckpt"
    model = betaTCVAE_ResNet.load_from_checkpoint(path_ckpt)
    model.eval()  # don't sample from latent: use mean


    log.info("Visualizing Latent Space")
    viz = Visualizer(model=model,
                     model_dir=output_dir,
                     dataloader=datamodule,
                     input_dim = config.evaluation.input_dim,
                     latent_dim=config.evaluation.latent_dim,
                     max_traversal=config.evaluation.max_traversal,
                     upsample_factor=config.evaluation.upsample_factor)
                     
    # same samples for all plots: sample max then take first `x`data  for all plots
    num_samples = config.evaluation.latent_samples * config.evaluation.latent_dim
    samples = get_samples(datamodule, num_samples)


    viz.traversals(data=samples[0:1, ...] if config.evaluation.is_posterior else None,
                    n_per_latent=config.evaluation.latent_samples,
                    n_latents=config.evaluation.latent_dim)

    viz.reconstruct_traverse(samples,
                                is_posterior=config.evaluation.is_posterior,
                                n_latents=config.evaluation.latent_dim,
                                n_per_latent=config.evaluation.latent_samples)

    viz.gif_traversals(samples[:5, ...], n_latents=config.evaluation.latent_dim, n_per_gif=60)
    
    log.info("Done!")