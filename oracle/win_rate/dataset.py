from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split


def split_and_shuffle_dataset(
    path_to_numpy_dataset: Path,
    directory_for_output: Path,
) -> None:
    """Quick method to open and split the large design matrix into
    train, val, test.
    """
    dset = np.load(path_to_numpy_dataset, allow_pickle=True)
    X, y, feature_names = dset["X"], dset["y"], dset["feature_names"]

    n = len(X)
    assert n == len(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.3, shuffle=True)

    # ugly but this is too much data to return to memory so save locally first
    np.savez(
        directory_for_output / "train",
        X=X_train,
        y=y_train,
        feature_names=feature_names,
        allow_pickle=True,
    )
    np.savez(
        directory_for_output / "test",
        X=X_test,
        y=y_test,
        feature_names=feature_names,
        allow_pickle=True,
    )
    np.savez(
        directory_for_output / "val",
        X=X_val,
        y=y_val,
        feature_names=feature_names,
        allow_pickle=True,
    )


class DatasetLoader:
    def __init__(
        self,
        directory_to_data: Path,
    ) -> None:
        self.directory_to_data = directory_to_data

    def _subsample(
        self, X: np.ndarray, y: np.ndarray, subsample_probability: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(X)
        indicies = np.random.choice(n, int(subsample_probability * n))
        return X[indicies], y[indicies]

    def load(
        self,
        dataset: str = "train",
        subsample_probability: Optional[float] = None,
        return_variable_names: Optional[bool] = False,
    ) -> Dict[str, np.ndarray]:
        """Cheap method to load in and subsample the data we would like from disk"""

        dataset = np.load(self.directory_to_data / f"{dataset}.npz")
        X, y = dataset["X"], dataset["y"]

        if subsample_probability is not None:
            X, y = self._subsample(X, y, subsample_probability)

        data = {"X": X, "y": y}
        if return_variable_names:
            data.update({"feature_names": dataset["feature_names"]})

        return data
