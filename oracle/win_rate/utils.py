import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _sum = np.sum(y_true == y_pred).item()
    return _sum / len(y_true)
