from argparse import ArgumentParser
from pathlib import Path

from oracle.utils.data import CONFIG_DIRECTORY, DATA_DIRECTORY
from oracle.utils.logger import getLogger
from oracle.win_rate.dataset import DatasetLoader
from oracle.win_rate.models.base import WinRateModel, WinRateModelConfig

logger = getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = ArgumentParser(description="Script to train win-rate model with basic structure.")
    parser.add_argument(
        "--path_to_model_config_yaml",
        type=Path,
        required=False,
        default=CONFIG_DIRECTORY / "wr_model_training.yaml",
    )
    parser.add_argument(
        "--path_to_dataset_directory", type=Path, required=False, default=DATA_DIRECTORY
    )
    args = parser.parse_args()
    return args


def main(
    path_to_model_config_yaml: Path,
    path_to_dataset_directory: Path,
):
    """Main function for fitting final model.

    Sequentially loads training, validation and test datasets given memory constraints."""
    # create model
    config = WinRateModelConfig.load_from_yaml(path_to_model_config_yaml)
    model = WinRateModel(config)
    logger.debug("initiated model")

    dataloader = DatasetLoader(path_to_dataset_directory)

    # all the training and validation all data at once takes too much memory
    train = dataloader.load("train")

    # fit
    model.fit(X=train["X"], y=train["y"])
    logger.debug("fit model")
    train_accuracy = model.evaluate(train["X"], train["y"])

    # validation and test
    del train
    val = dataloader.load("val")
    validation_accuracy = model.evaluate(val["X"], val["y"])

    del val
    test = dataloader.load("test")
    test_accuracy = model.evaluate(test["X"], test["y"])

    logger.info(
        f"Evaluation scores: train: {train_accuracy:.3f}, val: {validation_accuracy:.3f}, test: {test_accuracy:.3f}"
    )

    # save
    model.save_model()
    logger.debug(f"saved model to {model.get_save_path()}")


if __name__ == "__main__":
    args = parse_args()
    main(
        args.path_to_model_config_yaml,
        args.path_to_dataset_directory,
    )
