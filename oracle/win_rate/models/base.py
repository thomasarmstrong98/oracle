from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, TensorDataset

import oracle
from oracle.utils.data import DATA_DIRECTORY
from oracle.utils.logger import getLogger
from oracle.win_rate.features import BasicFeatureGenerator
from oracle.win_rate.utils import accuracy_score


class DeepNetModel(nn.Module):
    def __init__(self, feature_number: int) -> None:
        super().__init__()
        self.feature_number = feature_number

        self.l1 = nn.Sequential(nn.Linear(self.feature_number, 5_000), nn.Tanh())
        self.l2 = nn.Sequential(nn.Linear(5_000, 3_000), nn.LeakyReLU(negative_slope=0.2))
        # self.l3 = nn.Sequential(
        #     nn.Linear(3_000, 75),
        #     nn.LeakyReLU(negative_slope=0.2),
        # )
        self.final = nn.Linear(5_000 + 3_000, 1)

    def forward(self, x):
        l1 = self.l1(x)
        l2 = self.l2(l1)
        # l3 = self.l3(l2)

        out = self.final(torch.concat((l1, l2), axis=1)).flatten()
        return out


class LogisticRegressionModel(nn.Module):
    def __init__(self, feature_number: int) -> None:
        super().__init__()
        self.feature_number = feature_number
        self.l = nn.Linear(self.feature_number, 1)

    def forward(self, x: torch.TensorType) -> torch.TensorType:
        return self.l(x).flatten()


@dataclass
class WinRateModelConfig:
    # model config
    input_feature_dimensions: int = 126  # (draft encoding + embeddings)

    # operational configs
    use_gpu: bool = True
    model_save_directory: Path = DATA_DIRECTORY / "models"
    name: str = "win-rate-model"
    run_number: int = 0

    # training/validation configs
    binary_training_objective: str = "cross_entropy"
    epochs: int = 3
    early_stopping_rounds: int = 3
    training_batch_size: int = 100
    validation_batch_size: int = 1_000
    learning_rate: float = 1e-3
    weight_decay_coef: float = 1e-3
    evaluation_function: str = "accuracy_score"

    @classmethod
    def load_from_yaml(cls, path_to_yaml: Path):
        """Convenience method to instantiate from yaml file."""
        with open(path_to_yaml, "r") as stream:
            loaded_config = yaml.safe_load(stream)

        processors = {
            "model_save_directory": lambda x: Path(oracle.__file__).parents[1] / "data" / x,
        }

        processed_configs = {
            var: processors[var](value) for var, value in loaded_config.items() if var in processors
        }
        loaded_config.update(processed_configs)

        return cls(**loaded_config)


class WinRateModel:
    """Base Model for DOTA2 draft win classification"""

    def __init__(self, config: WinRateModelConfig):
        self.config = config
        self.logger = getLogger(self.config.name)

        self.device = self.get_device()

        self.model = DeepNetModel(self.config.input_feature_dimensions)
        self.to_device()

    def to_device(self):
        self.logger.debug(f"Assigned model to device {self.device}")
        self.model.to(self.device)

    def get_device(self):
        if self.config.use_gpu and torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"

        self.logger.debug(f"Found device {device}")
        return torch.device(device)

    def fit(self, X, y, X_val=None, y_val=None):
        validation = X_val is not None
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)

        X = torch.tensor(X).float()
        y = torch.tensor(y).float()

        if self.config.binary_training_objective == "negative_log_likelihood":
            loss_func = nn.NLLLoss()
        else:
            loss_func = nn.CrossEntropyLoss()

        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.config.training_batch_size,
            shuffle=True,
            num_workers=4,
        )

        if validation:
            X_val, y_val = torch.tensor(X_val).float(), torch.tensor(y_val).float()
            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=self.config.validation_batch_size,
                shuffle=True,
            )

        min_val_loss = float("inf")
        min_val_loss_idx = 0

        loss_history = []
        val_loss_history = []

        self.logger.info(
            f"Fitting model with {self.config.epochs} epochs, "
            f"batch size of {self.config.training_batch_size} for "
            f"a dataset of {X.shape[0]} observations"
        )

        for epoch in range(self.config.epochs):
            for i, (batch_X, batch_y) in enumerate(train_loader):

                out = self.model(batch_X.to(self.device))

                loss = loss_func(out, batch_y.to(self.device))
                loss_history.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if validation:
                # Early Stopping
                val_loss = 0.0
                val_dim = 0
                for val_i, (batch_val_X, batch_val_y) in enumerate(val_loader):
                    out = self.model(batch_val_X.to(self.device))

                    val_loss += loss_func(out, batch_val_y.to(self.device))
                    val_dim += 1

                val_loss /= val_dim
                val_loss_history.append(val_loss.item())

                self.logger.debug(f"Epoch {epoch}: Validation Loss: {val_loss:.5f}")

                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    min_val_loss_idx = epoch

                    # Save the currently best model
                    self.save_model("best")

                if min_val_loss_idx + self.config.early_stopping_rounds < epoch:
                    self.logger.info(
                        f"Validation loss has not improved for {self.config.early_stopping_rounds} steps."
                    )
                    self.logger.info("Applying early stopping")
                    break
        if validation:
            # Load best model
            self.load_model(filename="best")

        return loss_history, val_loss_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.shape[0] == 1:
            # single draft observation, use more peformant inference
            prediction = self.predict_proba_single_observation(X)
        else:
            prediction = self.predict_proba(X)

        return 2 * ((prediction > 0.5).astype(np.float16)) - 1.0

    def probabilistic_predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        # randomly simulate if we assign 0/1 based on probability.
        random_samples = np.random.rand(probabilities.shape[0])
        predictions = (random_samples < probabilities).astype(np.int0)
        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = self.predict_helper(X)

        return probas

    def predict_proba_single_observation(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            preds = self.model(torch.Tensor(X).to(self.device))
            preds = torch.sigmoid(preds)
            return preds.detach().cpu().numpy()

    def predict_helper(self, X: np.ndarray):
        self.model.eval()

        X = torch.tensor(X).float()
        test_dataset = TensorDataset(X)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.config.validation_batch_size,
            shuffle=False,
            num_workers=5,
        )
        predictions = []
        with torch.no_grad():
            for batch_X in test_loader:
                preds = self.model(batch_X[0].to(self.device))
                preds = torch.sigmoid(preds)  # binary classification
                predictions.append(preds.detach().cpu().numpy())

        return np.concatenate(predictions)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        if self.config.evaluation_function != "accuracy_score":
            raise NotImplementedError("Missing this evaluation function.")

        predictions = self.predict(X)
        eval_score = accuracy_score(y, predictions)

        return eval_score

    def save_model(self, filename: str = "win_rate_model") -> None:
        path = self.get_save_path(filename)
        torch.save(self.model.state_dict(), path)

    def load_model(self, filename: str = "win_rate_model") -> None:
        path = self.get_save_path(filename)
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)

    def get_save_path(self, filename: str = "win_rate_model") -> Path:
        return Path(
            self.config.model_save_directory
            / f"{self.config.name}_{self.config.run_number}_{filename}.model"
        )

    def get_model_size(self) -> float:
        model_size = sum(t.numel() for t in self.model.parameters() if t.requires_grad)
        return model_size

    def clone(self):
        """Clone the model.
        Creates a fresh copy of the model using the same parameters, but ignoring any trained weights. This method
        is necessary for the cross validation.
        :return: Copy of the current model without trained parameters
        """
        return self.__class__(self.config)


class WinRateClassificationWrapper:
    def __init__(
        self, feature_generator: BasicFeatureGenerator, win_rate_model: WinRateModel
    ) -> None:
        self.feature_generator = feature_generator
        self.win_rate_model = win_rate_model

    def __call__(self, draft: np.ndarray) -> np.ndarray:
        _, features = self.feature_generator(draft)
        classification = self.win_rate_model.predict(features)
        return classification
