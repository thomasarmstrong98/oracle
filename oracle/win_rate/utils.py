import torch


def accuracy_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    _sum = torch.sum(y_true == y_pred).item()
    return _sum / len(y_true)
