import hydra
from omegaconf import DictConfig


@hydra.main(config_path="configs/", config_name="config_eval.yaml")
def main(config: DictConfig):

    from src.evaluate import evaluate

    return evaluate(config)


if __name__ == "__main__":
    main()
