import hydra

from omegaconf import DictConfig


@hydra.main(config_path="configs/", config_name="config_tcvae.yaml")
def main(config: DictConfig):
    """Initializes utility loading and beta-TCVAE training within the hydra.main function,
    which applies the default beta-TCVAE training configuration.

    Parameters
    ----------
    config : omegaconf.DictConfig
        Configuration composed by Hydra.
    """

    from src.train import train
    from src.utils import utils

    utils.extras(config)

    return train(config)


if __name__ == "__main__":
    main()
