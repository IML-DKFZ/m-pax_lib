import os
from typing import List, Optional
import warnings

import hydra
from omegaconf import DictConfig

from pytorch_lightning import (
    LightningDataModule,
    seed_everything,
)

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from src.models.tcvae_resnet import betaTCVAE_ResNet
from src.models.tcvae_conv import betaTCVAE_Conv
from src.models.head_mlp import MLP

from src.evaluation.vis_LS import *
from src.evaluation.vis_AM import *

from src.utils import utils

log = utils.get_logger(__name__)

warnings.filterwarnings("ignore")


def evaluate(config: DictConfig) -> Optional[float]:

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

    log.info("Visualizing Latent Space")
    vis = Visualizer(
        model=encoder,
        model_dir=output_dir,
        dataloader=datamodule,
        input_dim=config.evaluation.input_dim,
        latent_dim=encoder.state_dict()["fc_mu.weight"].shape[0],
        max_traversal=config.evaluation.max_traversal,
        upsample_factor=config.evaluation.upsample_factor,
        index=config.evaluation.index,
    )

    vis.reconstruct_traverse(
        dataloader=datamodule.train_dataloader(),
        is_posterior=config.evaluation.is_posterior,
        n_latents=encoder.state_dict()["fc_mu.weight"].shape[0],
        n_per_latent=config.evaluation.latent_samples,
    )

    vis.gif_traversals(
        dataloader=datamodule.train_dataloader(),
        n_latents=encoder.state_dict()["fc_mu.weight"].shape[0],
        n_per_gif=60,
        index=config.evaluation.index,
    )

    log.info("Computing Attribution of:")
    log.info("original -> output (1/3)")

    AttributionOriginalY(
        head=head,
        dataloader=datamodule.train_dataloader(),
        index=config.evaluation.index,
        output_dir=output_dir,
        dataset=datamodule.name,
        baseline=config.evaluation.baseline,
        kernel_size=config.evaluation.kernel_size,
    ).visualization()

    log.info("latent -> output (2/3)")

    AttributionLatentY(
        head=head,
        encoder=encoder,
        dataloader=datamodule.train_dataloader(),
        output_dir=output_dir,
        index=config.evaluation.index,
        dataset=datamodule.name,
    ).visualization()

    log.info("original -> latent (3/3)")

    AttributionOriginalLatent(
        encoder=encoder,
        dataloader=datamodule.train_dataloader(),
        index=config.evaluation.index,
        latent_dim=encoder.state_dict()["fc_mu.weight"].shape[0],
        output_dir=output_dir,
        baseline=config.evaluation.baseline,
        kernel_size=config.evaluation.kernel_size, 
    ).visualization()

    log.info("Done!")
