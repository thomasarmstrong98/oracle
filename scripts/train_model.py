"""Train DOTA2 draft win rate prediction model."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from oracle.config import ModelConfig, TrainingConfig
from oracle.models import DraftNet
from oracle.utils.data import create_draft, get_opendota_hero_embedding, read_draft_data
from oracle.utils.logger import getLogger

logger = getLogger(__name__)


def load_data(config: TrainingConfig):
    """Load and split data."""
    data_path = config.data_dir / "dota.pickle"
    draft_data = read_draft_data(data_path)

    # Create draft arrays
    drafts = create_draft(draft_data.radiant_draft, draft_data.dire_draft)
    labels = draft_data.radiant_win.astype(np.float32)

    # Get hero embeddings (will be used to initialize model)
    embeddings = get_opendota_hero_embedding().values.astype(np.float32)

    # Create features (draft encoding + aggregated embeddings for compatibility)
    features = np.concatenate([drafts, drafts @ embeddings], axis=1)

    # Split data
    n = len(features)
    train_idx = int(n * config.train_split)
    val_idx = int(n * (config.train_split + config.val_split))

    splits = {
        "train": (features[:train_idx], labels[:train_idx]),
        "val": (features[train_idx:val_idx], labels[train_idx:val_idx]),
        "test": (features[val_idx:], labels[val_idx:]),
    }

    logger.info(f"Loaded {n} samples")
    logger.info(
        f"Train: {len(splits['train'][0])}, Val: {len(splits['val'][0])}, Test: {len(splits['test'][0])}"
    )
    logger.info(f"Feature dim: {features.shape[1]}")

    return splits, embeddings


def create_dataloader(X, y, batch_size, shuffle=True):
    """Create a PyTorch DataLoader."""
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True
    )


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds, all_targets = [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(X).squeeze()
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            preds = torch.sigmoid(logits) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    return total_loss / len(loader), acc


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_preds, all_probs, all_targets = [], [], []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)

            logits = model(X).squeeze()
            loss = criterion(logits, y)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = probs > 0.5

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    auc = roc_auc_score(all_targets, all_probs)

    return total_loss / len(loader), acc, f1, auc


def train_model(model_config: ModelConfig, train_config: TrainingConfig):
    """Main training loop."""
    # Set seed
    torch.manual_seed(train_config.seed)
    np.random.seed(train_config.seed)

    # Device
    if train_config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(train_config.device)
    logger.info(f"Using device: {device}")

    # Load data
    data, embeddings = load_data(train_config)
    train_loader = create_dataloader(*data["train"], train_config.batch_size, shuffle=True)
    val_loader = create_dataloader(*data["val"], train_config.batch_size, shuffle=False)
    test_loader = create_dataloader(*data["test"], train_config.batch_size, shuffle=False)

    # Create model
    model = DraftNet(
        num_heroes=model_config.num_heroes,
        embedding_dim=model_config.embedding_dim,
        hero_hidden_dim=model_config.hero_hidden_dim,
        num_attention_heads=model_config.num_attention_heads,
        draft_hidden_dim=model_config.draft_hidden_dim,
        dropout=model_config.dropout,
    ).to(device)

    # Initialize hero embeddings
    model.initialize_embeddings(torch.from_numpy(embeddings).float().to(device))

    logger.info(f"Model: {model}")
    logger.info(f"Parameters: {model.get_num_parameters():,}")

    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay
    )

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0

    logger.info("Starting training")
    for epoch in tqdm(range(train_config.epochs), desc="Training"):
        start_time = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1, val_auc = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - start_time

        logger.info(f"Epoch {epoch+1}/{train_config.epochs} ({epoch_time:.1f}s)")
        logger.info(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        logger.info(
            f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss - train_config.min_delta:
            best_val_loss = val_loss
            patience_counter = 0

            checkpoint_path = train_config.checkpoint_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
            logger.debug(f"  Saved checkpoint: {checkpoint_path}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= train_config.patience:
            logger.info(f"Early stopping after {epoch+1} epochs")
            break

    # Load best model and evaluate on test set
    logger.info("Loading best model for test evaluation")
    checkpoint = torch.load(train_config.checkpoint_dir / "best_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, test_f1, test_auc = validate(model, test_loader, criterion, device)
    logger.info("Test Results:")
    logger.info(f"  Loss: {test_loss:.4f}")
    logger.info(f"  Accuracy: {test_acc:.4f}")
    logger.info(f"  F1 Score: {test_f1:.4f}")
    logger.info(f"  AUC-ROC: {test_auc:.4f}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train DOTA2 draft win rate model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    args = parser.parse_args()

    model_config = ModelConfig()
    train_config = TrainingConfig(
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, device=args.device
    )

    train_model(model_config, train_config)


if __name__ == "__main__":
    main()
