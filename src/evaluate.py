import os
from typing import List, Optional

import hydra
from omegaconf import DictConfig

from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    seed_everything,
)
import torch

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from src.models.tcvae_resnet import betaTCVAE_ResNet
from src.models.head_mlp import MLP

from src.evaluation.vis_LS import *
from src.evaluation.vis_AM import *
from src.evaluation.scores_AM import *

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
    model.eval()

    path_ckpt = config.data_dir + "models/" + config.evaluation.model_dir + "/head.ckpt"
    head = MLP.load_from_checkpoint(path_ckpt)
    model.eval()


    log.info("Visualizing Latent Space")
    vis = Visualizer(model=model,
                     model_dir=output_dir,
                     dataloader=datamodule,
                     input_dim = config.evaluation.input_dim,
                     latent_dim=model.state_dict()['fc_mu.weight'].shape[0],
                     max_traversal=config.evaluation.max_traversal,
                     upsample_factor=config.evaluation.upsample_factor)
                     
    # same samples for all plots: sample max then take first `x`data  for all plots
    num_samples = config.evaluation.latent_samples * model.state_dict()['fc_mu.weight'].shape[0]
    samples = get_samples(datamodule, num_samples)


    # vis.traversals(data=samples[0:1, ...] if config.evaluation.is_posterior else None,
    #                 n_per_latent=config.evaluation.latent_samples,
    #                 n_latents=cmodel.state_dict()['fc_mu.weight'].shape[0])

    vis.reconstruct_traverse(samples,
                                is_posterior=config.evaluation.is_posterior,
                                n_latents=model.state_dict()['fc_mu.weight'].shape[0],
                                n_per_latent=config.evaluation.latent_samples)

    vis.gif_traversals(samples[:5, ...], n_latents=model.state_dict()['fc_mu.weight'].shape[0], n_per_gif=60)

    log.info("Computing Attribution")
    log.info("original -> output (1/3)")
    scores_original, test_images_original = scores_AM_Original(head, 
                                            datamodule.train_dataloader(),
                                            method = config.evaluation.method,
                                            out_dim = head.state_dict()['fc2.weight'].shape[0]
                                            ).compute()

    log.info("latent -> output (2/3)")
    exp, scores_latent, encoding_test, labels_test = scores_AM_Latent(model = head,
                                            encoder = model,
                                            datamodule=datamodule.train_dataloader(),
                                            method = config.evaluation.method
                                            ).compute()

    log.info("original -> latent (3/3)")
    scores_oil, test_images_oil = scores_AM_Original(model,
                                        datamodule.train_dataloader(),
                                        method = config.evaluation.method,
                                        out_dim = model.state_dict()['fc_mu.weight'].shape[0]
                                        ).compute()

    log.info("Visualizing Attribution")
    vis_AM_Original(scores_original, test_images_original).visualise()
    plt.savefig(output_dir + 'attribution_original.png')

    vis_AM_Latent(shap_values=scores_latent,
                explainer=exp, 
                encoding_test=encoding_test,
                labels_test=labels_test,
                output_dir=output_dir
                ).visualise()

    vis_AM_Original(scores_oil, test_images_oil).visualise()
    plt.savefig(output_dir + 'attribution_original_into_LSF.png')

    log.info("Done!")