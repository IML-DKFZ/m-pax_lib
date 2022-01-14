import hydra

from omegaconf import DictConfig


@hydra.main(config_path="configs/", config_name="config_eval.yaml")
def main(config: DictConfig):
    """Initializes model evaluation within the hydra.main function,
    which applies the default evaluation configuration.

    Parameters
    ----------
    config : omegaconf.DictConfig
        Configuration composed by Hydra.
    """

    from src.evaluate import evaluate

    return evaluate(config)


if __name__ == "__main__":
    main()
