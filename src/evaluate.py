import os
import warnings

import hydra

from omegaconf import DictConfig
from PIL import ImageFile
from pytorch_lightning import LightningDataModule, seed_everything
from typing import List, Optional

from src.evaluation.vis_AM import *
from src.evaluation.vis_LS import *
from src.models.head_mlp import MLP
from src.models.tcvae_conv import betaTCVAE_Conv
from src.models.tcvae_resnet import betaTCVAE_ResNet
from src.utils import utils

ImageFile.LOAD_TRUNCATED_IMAGES = True
log = utils.get_logger(__name__)
warnings.filterwarnings("ignore")


def evaluate(config: DictConfig):
    """Contains evaluation pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Parameters
    ----------
    config : omegaconf.DictConfig
        Configuration composed by Hydra.
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

    path_ckpt = (
        config.data_dir + "models/" + config.evaluation.model_dir + "/encoder.ckpt"
    )

    for architecture in [betaTCVAE_ResNet, betaTCVAE_Conv]:
        try:
            encoder = architecture.load_from_checkpoint(path_ckpt)
            break
        except RuntimeError:
            # repeat the loop on failure
            continue
    encoder.eval()

    path_ckpt = config.data_dir + "models/" + config.evaluation.model_dir + "/head.ckpt"
    head = MLP.load_from_checkpoint(path_ckpt)
    head.eval()

    # Init visualizer
    log.info("Visualizing Latent Space")
    vis = Visualizer(
        index=config.evaluation.index,
        latent_dim=encoder.state_dict()["fc_mu.weight"].shape[0],
        max_traversal=config.evaluation.max_traversal,
        model=encoder,
        model_dir=output_dir,
    )

    # Visualizing latent space
    vis.reconstruct_traverse(
        dataloader=datamodule.train_dataloader(),
        is_posterior=config.evaluation.is_posterior,
        n_latents=encoder.state_dict()["fc_mu.weight"].shape[0],
        n_per_latent=config.evaluation.latent_samples,
    )

    vis.gif_traversals(
        dataloader=datamodule.train_dataloader(),
        index=config.evaluation.index,
        n_latents=encoder.state_dict()["fc_mu.weight"].shape[0],
        n_per_gif=60,
    )

    # Computing the multi-path attributions
    log.info("Computing Attribution of:")
    log.info("original -> output (1/3)")

    AttributionOriginalY(
        baseline=config.evaluation.baseline,
        dataloader=datamodule.train_dataloader(),
        dataset=datamodule.name,
        head=head,
        index=config.evaluation.index,
        kernel_size=config.evaluation.kernel_size,
        output_dir=output_dir,
    ).visualization()

    log.info("latent -> output (2/3)")

    AttributionLatentY(
        dataloader=datamodule.train_dataloader(),
        dataset=datamodule.name,
        encoder=encoder,
        head=head,
        index=config.evaluation.index,
        output_dir=output_dir,
    ).visualization()

    log.info("original -> latent (3/3)")

    AttributionOriginalLatent(
        baseline=config.evaluation.baseline,
        dataloader=datamodule.train_dataloader(),
        encoder=encoder,
        index=config.evaluation.index,
        kernel_size=config.evaluation.kernel_size,
        latent_dim=encoder.state_dict()["fc_mu.weight"].shape[0],
        output_dir=output_dir,
    ).visualization()

    log.info("Done!")
