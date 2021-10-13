import hydra
from omegaconf import DictConfig


@hydra.main(config_path="configs/", config_name="config_head.yaml")
def main(config: DictConfig):

    from src.train import train
    from src.utils import utils

    # A couple of optional utilities:
    # - disabling python warnings
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # You can safely get rid of this line if you don't want those
    utils.extras(config)

    # Pretty print config using Rich library
    # if config.get("print_config"):
    #    utils.print_config(config, resolve=True)

    # Train model
    return train(config)


if __name__ == "__main__":
    main()
