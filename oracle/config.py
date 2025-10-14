"""Configuration for Oracle training."""

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class ModelConfig:
    num_heroes: int = 112
    embedding_dim: int = 14
    hero_hidden_dim: int = 128
    num_attention_heads: int = 4
    draft_hidden_dim: int = 256
    dropout: float = 0.2


@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    patience: int = 15
    min_delta: float = 1e-4

    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15

    device: str = "auto"
    seed: int = 42

    data_dir: Path = Path("data")
    checkpoint_dir: Path = Path("checkpoints")

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)


@dataclass
class AlphaZeroConfig:
    """Configuration for AlphaZero self-play training."""

    # Model config
    num_heroes: int = 112
    embedding_dim: int = 14
    hero_hidden_dim: int = 128
    num_attention_heads: int = 4
    draft_hidden_dim: int = 256
    dropout: float = 0.2

    # Training config
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    batch_size: int = 256

    # Self-play config
    num_selfplay_iterations: int = 100
    num_games_per_iteration: int = 100
    num_simulations: int = 800
    cpuct: float = 1.0
    tree_reset_interval: int = 10

    # MCTS config
    search_time_seconds: int = 60
    temperature: float = 1.0  # For sampling during self-play

    # System
    device: str = "auto"
    seed: int = 42
    num_workers: int = 4

    data_dir: Path = Path("data")
    checkpoint_dir: Path = Path("checkpoints/alphazero")

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
