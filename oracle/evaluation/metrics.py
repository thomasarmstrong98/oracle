"""Evaluation metrics for AlphaZero models."""

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score


def calculate_policy_accuracy(
    predicted_policies: np.ndarray,
    target_policies: np.ndarray,
    k: int = 5
) -> float:
    """Calculate top-k policy accuracy.

    Args:
        predicted_policies: Model policy predictions (batch, num_actions)
        target_policies: Target policy distributions (batch, num_actions)
        k: Top-k to consider

    Returns:
        Top-k accuracy: fraction of times true best action is in top-k predictions
    """
    # Get top-k predicted actions
    top_k_pred = np.argsort(predicted_policies, axis=1)[:, -k:]

    # Get true best action
    true_best = np.argmax(target_policies, axis=1)

    # Check if true best is in top-k
    accuracy = np.mean([true_best[i] in top_k_pred[i] for i in range(len(true_best))])

    return accuracy


def calculate_value_calibration(
    predicted_values: np.ndarray,
    actual_outcomes: np.ndarray,
    num_bins: int = 10
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Calculate value prediction calibration (Expected Calibration Error).

    Args:
        predicted_values: Model value predictions in [-1, 1] (batch,)
        actual_outcomes: Actual game outcomes in [-1, 1] (batch,)
        num_bins: Number of calibration bins

    Returns:
        ece: Expected Calibration Error
        bin_accuracies: Accuracy per bin
        bin_confidences: Average confidence per bin
    """
    # Convert values to [0, 1] for easier binning
    pred_probs = (predicted_values + 1) / 2
    true_probs = (actual_outcomes + 1) / 2

    # Create bins
    bins = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(pred_probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    # Calculate per-bin statistics
    bin_accuracies = np.zeros(num_bins)
    bin_confidences = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)

    for i in range(num_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_confidences[i] = pred_probs[mask].mean()
            bin_accuracies[i] = true_probs[mask].mean()
            bin_counts[i] = mask.sum()

    # Calculate ECE (Expected Calibration Error)
    ece = np.sum(bin_counts * np.abs(bin_accuracies - bin_confidences)) / len(predicted_values)

    return ece, bin_accuracies, bin_confidences


def calculate_policy_entropy(policies: np.ndarray, valid_masks: np.ndarray = None) -> float:
    """Calculate average policy entropy.

    Higher entropy means more exploratory/uncertain policies.

    Args:
        policies: Policy distributions (batch, num_actions)
        valid_masks: Binary masks for valid actions (batch, num_actions)

    Returns:
        Average entropy across batch
    """
    if valid_masks is not None:
        # Mask invalid actions
        policies = policies * valid_masks
        policies = policies / (policies.sum(axis=1, keepdims=True) + 1e-8)

    # Calculate entropy: -sum(p * log(p))
    entropy = -np.sum(policies * np.log(policies + 1e-8), axis=1)

    return entropy.mean()


def calculate_value_accuracy(
    predicted_values: np.ndarray,
    actual_outcomes: np.ndarray,
    threshold: float = 0.0
) -> Tuple[float, float, float]:
    """Calculate value prediction accuracy metrics.

    Args:
        predicted_values: Model value predictions in [-1, 1] (batch,)
        actual_outcomes: Actual game outcomes in [-1, 1] (batch,)
        threshold: Threshold for binary classification

    Returns:
        accuracy: Binary classification accuracy
        mse: Mean squared error
        mae: Mean absolute error
    """
    # Binary accuracy (sign prediction)
    pred_sign = (predicted_values > threshold).astype(int)
    true_sign = (actual_outcomes > threshold).astype(int)
    accuracy = accuracy_score(true_sign, pred_sign)

    # MSE and MAE
    mse = np.mean((predicted_values - actual_outcomes) ** 2)
    mae = np.mean(np.abs(predicted_values - actual_outcomes))

    return accuracy, mse, mae


def evaluate_model_on_examples(
    model,
    drafts: np.ndarray,
    target_policies: np.ndarray,
    target_values: np.ndarray,
    device: torch.device,
    batch_size: int = 512
) -> dict:
    """Comprehensive evaluation of model on a dataset.

    Args:
        model: AlphaZeroNet model
        drafts: Draft states (N, num_heroes)
        target_policies: Target policy distributions (N, num_heroes)
        target_values: Target values (N,)
        device: Torch device
        batch_size: Batch size for inference

    Returns:
        Dictionary of metrics
    """
    model.eval()

    all_pred_policies = []
    all_pred_values = []
    policy_losses = []
    value_losses = []

    with torch.no_grad():
        for i in range(0, len(drafts), batch_size):
            batch_drafts = torch.from_numpy(drafts[i:i+batch_size]).float().to(device)
            batch_target_policies = torch.from_numpy(target_policies[i:i+batch_size]).float().to(device)
            batch_target_values = torch.from_numpy(target_values[i:i+batch_size]).float().to(device)

            # Forward pass
            policy_logits, values = model(batch_drafts)

            # Calculate losses
            log_probs = F.log_softmax(policy_logits, dim=1)
            policy_loss = -(batch_target_policies * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(values.squeeze(), batch_target_values)

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

            # Get predictions
            policies = F.softmax(policy_logits, dim=1).cpu().numpy()
            values_np = values.squeeze().cpu().numpy()

            all_pred_policies.append(policies)
            all_pred_values.append(values_np)

    # Concatenate all predictions
    pred_policies = np.concatenate(all_pred_policies, axis=0)
    pred_values = np.concatenate(all_pred_values, axis=0)

    # Calculate metrics
    metrics = {
        'policy_loss': np.mean(policy_losses),
        'value_loss': np.mean(value_losses),
        'policy_top1_accuracy': calculate_policy_accuracy(pred_policies, target_policies, k=1),
        'policy_top5_accuracy': calculate_policy_accuracy(pred_policies, target_policies, k=5),
        'policy_entropy': calculate_policy_entropy(pred_policies),
        'value_sign_accuracy': calculate_value_accuracy(pred_values, target_values)[0],
        'value_mse': calculate_value_accuracy(pred_values, target_values)[1],
        'value_mae': calculate_value_accuracy(pred_values, target_values)[2],
    }

    # Calculate calibration
    ece, _, _ = calculate_value_calibration(pred_values, target_values)
    metrics['value_ece'] = ece

    return metrics
