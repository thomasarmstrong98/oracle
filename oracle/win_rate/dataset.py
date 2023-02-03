from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split


def split_and_shuffle_dataset(
    path_to_sparse_design_matrix: Path,
    path_to_match_outcomes: Path,
    directory_for_output: Path,
) -> None:
    """Quick method to open and split the large design matrix into
    train, val, test.
    """
    X = sparse.load_npz(path_to_sparse_design_matrix)
    y = np.load(path_to_match_outcomes)

    n = len(X)
    assert n == len(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, 0.3, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, 0.3, shuffle=True)

    # ugly but this is too much data to return to memory so save locally first
    sparse.save_npz(directory_for_output / "X_train", X_train)
    sparse.save_npz(directory_for_output / "X_test", X_test)
    sparse.save_npz(directory_for_output / "X_val", X_val)

    np.save(directory_for_output / "y_train", y_train)
    np.save(directory_for_output / "y_test", y_test)
    np.save(directory_for_output / "y_val", y_val)


@dataclass
class ModelDataset:
    """Object for storing data used to train some win-rate models."""

    X_train: sparse.csr_matrix
    y_train: np.ndarray

    X_val: sparse.csr_matrix
    y_val: np.ndarray

    X_test: Optional[sparse.csr_matrix] = None
    y_test: Optional[np.ndarray] = None


class DatasetLoader:
    def __init__(
        self,
        directory_to_data: Path,
        subsample: bool = True,
        subsample_probability: float = 0.2,
        return_test_data: bool = False,
    ) -> None:
        self.local_path_to_data = directory_to_data
        self.subsample = subsample
        self.subsample_prob = subsample_probability
        self.return_test_data = return_test_data

        # hardcoded local dir structure
        self.directory_structure = {
            "train": ("X_train.npz", "y_train.npy"),
            "val": ("X_val.npz", "y_val.npy"),
            "test": ("X_test.npz", "y_test.npy"),
        }

    def _subsample(
        self, X: sparse.csr_matrix, y: np.ndarray
    ) -> Tuple[sparse.csr_matrix, np.ndarray]:
        n = len(X)
        indicies = np.random.choice(n, int(self.subsample_prob * n))
        return X[indicies], y[indicies]

    def load(self) -> ModelDataset:
        """Cheap method to load in and subsample the data we would like from disk"""
        X_train = sparse.load_npz(self.local_path_to_data / "X_train.npz")
        y_train = np.load(self.local_path_to_data / "y_train")
        if self.subsample:
            X_train, y_train = self._subsample(X_train, y_train)

        X_val = sparse.load_npz(self.local_path_to_data / "X_val.npz")
        y_val = np.load(self.local_path_to_data / "y_val.npy")
        if self.subsample:
            X_val, y_val = self._subsample(X_val, y_val)

        if self.return_test_data:
            X_test = sparse.load_npz(self.local_path_to_data / "X_test.npz")
            y_test = np.load(self.local_path_to_data / "y_test.npy")
            if self.subsample:
                X_test, y_test = self._subsample(X_test, y_test)

            return ModelDataset(X_train, y_train, X_val, y_val, X_test, y_test)

        return ModelDataset(X_train, y_train, X_val, y_val)
