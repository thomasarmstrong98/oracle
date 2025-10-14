"""AlphaZero self-play training for DOTA2 draft recommendation.

This script implements the full AlphaZero training loop:
1. Self-play: Generate training examples using MCTS with current network
2. Train: Update network on recent self-play examples
3. Evaluate: Compare new network against previous best
4. Checkpoint: Save model if performance improves
"""

import argparse
import pickle
import time
from collections import deque
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from oracle.config import AlphaZeroConfig
from oracle.drafting.alpha_zero.training import Coach, SelfPlayExample
from oracle.drafting.alpha_zero.neural_net import create_alphazero_net, NeuralNetWrapper
from oracle.drafting.game import BasicDraft
from oracle.drafting.mcts import AlphaZeroMCTS
from oracle.utils.data import get_opendota_hero_embedding
from oracle.utils.logger import getLogger

logger = getLogger(__name__)


class AlphaZeroTrainer:
    """Manages AlphaZero self-play training loop."""

    def __init__(self, config: AlphaZeroConfig, use_tensorboard: bool = True):
        self.config = config
        self.device = self._get_device()

        # Training state
        self.iteration = 0
        self.training_examples = deque(maxlen=config.num_games_per_iteration * 5)

        # Initialize network
        self.model = self._create_model()
        self.net_wrapper = NeuralNetWrapper(self.model, device=self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Create game and MCTS
        self.game = self._create_game()
        self.mcts = self._create_mcts()

        # TensorBoard writer
        self.writer: Optional[SummaryWriter] = None
        if use_tensorboard:
            log_dir = Path("runs") / f"alphazero_{time.strftime('%Y%m%d_%H%M%S')}"
            self.writer = SummaryWriter(log_dir=str(log_dir))
            logger.info(f"TensorBoard logging to: {log_dir}")

        # Metrics
        self.metrics_history = {
            "iteration": [],
            "policy_loss": [],
            "value_loss": [],
            "total_loss": [],
            "examples_count": [],
            "avg_game_length": [],
        }

    def _get_device(self) -> torch.device:
        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)
        logger.info(f"Using device: {device}")
        return device

    def _create_model(self) -> torch.nn.Module:
        """Create and initialize AlphaZero network."""
        model = create_alphazero_net(
            num_heroes=self.config.num_heroes,
            embedding_dim=self.config.embedding_dim,
            hero_hidden_dim=self.config.hero_hidden_dim,
            num_attention_heads=self.config.num_attention_heads,
            draft_hidden_dim=self.config.draft_hidden_dim,
            dropout=self.config.dropout,
        )

        # Initialize embeddings
        try:
            embeddings = get_opendota_hero_embedding().values.astype(np.float32)
            model.initialize_embeddings(torch.from_numpy(embeddings))
            logger.info("Initialized hero embeddings from OpenDota data")
        except Exception as e:
            logger.warning(f"Could not load embeddings: {e}. Using random init.")

        logger.info(f"Created AlphaZeroNet with {model.get_num_parameters():,} parameters")
        return model

    def _create_game(self) -> BasicDraft:
        """Create draft game with neural network as reward model."""

        def net_reward_model(draft: np.ndarray) -> float:
            """Use current network for position evaluation."""
            _, value = self.net_wrapper.predict(draft)
            return value

        return BasicDraft(reward_model=net_reward_model)

    def _create_mcts(self) -> AlphaZeroMCTS:
        """Create MCTS with current neural network."""
        return AlphaZeroMCTS(
            drafting_game=self.game,
            game_net=self.net_wrapper.predict,
            search_stop_criterion="simulations",
            search_number_of_simulations=self.config.num_simulations,
            cpuct=self.config.cpuct,
        )

    def run_selfplay_games(self, num_games: int) -> List[SelfPlayExample]:
        """Generate training examples through self-play.

        Args:
            num_games: Number of self-play games to run

        Returns:
            List of training examples from all games
        """
        logger.info(f"Running {num_games} self-play games...")
        coach = Coach(
            draft_game=self.game,
            mcts=self.mcts,
            tree_reset_interval=self.config.tree_reset_interval,
        )

        all_examples = []
        game_lengths = []

        for game_idx in tqdm(range(num_games), desc="Self-play"):
            examples = coach.selfplay_round()
            all_examples.extend(examples)
            game_lengths.append(len(examples))

        avg_length = np.mean(game_lengths)
        logger.info(f"Generated {len(all_examples)} examples (avg game: {avg_length:.1f} moves)")

        return all_examples

    def train_network(self, examples: List[SelfPlayExample], epochs: int = 10) -> dict:
        """Train network on self-play examples.

        Args:
            examples: Training examples from self-play
            epochs: Number of training epochs

        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training on {len(examples)} examples for {epochs} epochs...")

        # Convert to numpy arrays
        _, drafts, policies, outcomes = Coach.convert_examples_to_numpy(examples)

        # Training loop
        policy_losses = []
        value_losses = []
        total_losses = []

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(len(drafts))

            epoch_policy_loss = []
            epoch_value_loss = []
            epoch_total_loss = []

            # Mini-batch training
            for i in range(0, len(drafts), self.config.batch_size):
                batch_indices = indices[i : i + self.config.batch_size]

                batch_drafts = drafts[batch_indices]
                batch_policies = policies[batch_indices]
                batch_outcomes = outcomes[batch_indices]

                total_loss, policy_loss, value_loss = self.net_wrapper.train_on_batch(
                    batch_drafts,
                    batch_policies,
                    batch_outcomes,
                    self.optimizer,
                    policy_weight=self.config.policy_loss_weight,
                    value_weight=self.config.value_loss_weight,
                )

                epoch_policy_loss.append(policy_loss)
                epoch_value_loss.append(value_loss)
                epoch_total_loss.append(total_loss)

            policy_losses.append(np.mean(epoch_policy_loss))
            value_losses.append(np.mean(epoch_value_loss))
            total_losses.append(np.mean(epoch_total_loss))

        metrics = {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "total_loss": np.mean(total_losses),
        }

        logger.info(
            f"Training complete - "
            f"Policy: {metrics['policy_loss']:.4f}, "
            f"Value: {metrics['value_loss']:.4f}, "
            f"Total: {metrics['total_loss']:.4f}"
        )

        return metrics

    def save_checkpoint(self, iteration: int, is_best: bool = False):
        """Save model checkpoint with full training state.

        Args:
            iteration: Current training iteration
            is_best: Whether this is the best model so far
        """
        checkpoint_data = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'iteration': iteration,
            'training_examples': list(self.training_examples),
            'metrics_history': self.metrics_history,
            'config': self.config,
        }

        # Save periodic checkpoint
        checkpoint_path = self.config.checkpoint_dir / f"checkpoint_iter_{iteration}.pt"
        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save as best if applicable
        if is_best:
            best_path = self.config.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint_data, best_path)
            logger.info(f"Saved best model: {best_path}")

        # Save training history separately for easy access
        history_path = self.config.checkpoint_dir / "training_history.pkl"
        with open(history_path, "wb") as f:
            pickle.dump(self.metrics_history, f)

        # Keep only last 3 checkpoints to save space
        self._cleanup_old_checkpoints(keep_last=3)

    def _cleanup_old_checkpoints(self, keep_last: int = 3):
        """Remove old checkpoints, keeping only the most recent ones.

        Args:
            keep_last: Number of recent checkpoints to keep
        """
        checkpoint_pattern = self.config.checkpoint_dir / "checkpoint_iter_*.pt"
        checkpoints = sorted(self.config.checkpoint_dir.glob("checkpoint_iter_*.pt"))

        if len(checkpoints) > keep_last:
            for checkpoint in checkpoints[:-keep_last]:
                checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {checkpoint}")

    def load_checkpoint(self, checkpoint_path: Path):
        """Load checkpoint and resume training.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.iteration = checkpoint['iteration']
        self.training_examples = deque(checkpoint['training_examples'], maxlen=self.config.num_games_per_iteration * 5)
        self.metrics_history = checkpoint['metrics_history']

        logger.info(f"Resumed from iteration {self.iteration}")
        logger.info(f"Training examples buffer: {len(self.training_examples)}")

    def train(self, num_iterations: int):
        """Run full AlphaZero training loop.

        Args:
            num_iterations: Number of self-play → train iterations
        """
        logger.info("=" * 60)
        logger.info("Starting AlphaZero Training")
        logger.info("=" * 60)
        logger.info(f"Iterations: {num_iterations}")
        logger.info(f"Games per iteration: {self.config.num_games_per_iteration}")
        logger.info(f"MCTS simulations: {self.config.num_simulations}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info("=" * 60)

        best_loss = float("inf")

        for iteration in range(1, num_iterations + 1):
            self.iteration = iteration
            iter_start_time = time.time()

            logger.info(f"\n{'='*60}")
            logger.info(f"Iteration {iteration}/{num_iterations}")
            logger.info(f"{'='*60}")

            # 1. Self-play
            examples = self.run_selfplay_games(self.config.num_games_per_iteration)
            self.training_examples.extend(examples)

            # 2. Train network
            metrics = self.train_network(list(self.training_examples), epochs=10)

            # 3. Track metrics
            avg_game_length = len(examples) / self.config.num_games_per_iteration
            self.metrics_history["iteration"].append(iteration)
            self.metrics_history["policy_loss"].append(metrics["policy_loss"])
            self.metrics_history["value_loss"].append(metrics["value_loss"])
            self.metrics_history["total_loss"].append(metrics["total_loss"])
            self.metrics_history["examples_count"].append(len(self.training_examples))
            self.metrics_history["avg_game_length"].append(avg_game_length)

            # Log to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar("Loss/policy", metrics["policy_loss"], iteration)
                self.writer.add_scalar("Loss/value", metrics["value_loss"], iteration)
                self.writer.add_scalar("Loss/total", metrics["total_loss"], iteration)
                self.writer.add_scalar("Training/examples_count", len(self.training_examples), iteration)
                self.writer.add_scalar("Training/avg_game_length", avg_game_length, iteration)
                self.writer.add_scalar("Training/iteration_time", time.time() - iter_start_time, iteration)
                self.writer.add_scalar("Training/learning_rate", self.config.learning_rate, iteration)

            # 4. Checkpoint
            is_best = metrics["total_loss"] < best_loss
            if is_best:
                best_loss = metrics["total_loss"]

            if iteration % 5 == 0 or is_best:
                self.save_checkpoint(iteration, is_best=is_best)

            iter_time = time.time() - iter_start_time
            logger.info(f"Iteration time: {iter_time:.1f}s")
            logger.info(f"Training examples buffer: {len(self.training_examples)}")

        logger.info("\n" + "=" * 60)
        logger.info("Training Complete!")
        logger.info("=" * 60)
        logger.info(f"Best loss: {best_loss:.4f}")
        logger.info(f"Final model: {self.config.checkpoint_dir / 'best_model.pt'}")

        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()
            logger.info("TensorBoard logging complete")


def main():
    parser = argparse.ArgumentParser(
        description="Train AlphaZero for DOTA2 draft recommendation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training parameters
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of self-play training iterations",
    )
    parser.add_argument(
        "--games-per-iter",
        type=int,
        default=25,
        help="Number of self-play games per iteration",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=400,
        help="MCTS simulations per move",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size",
    )

    # Model parameters
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--cpuct", type=float, default=1.0, help="MCTS exploration constant")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")

    # System
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/alphazero"),
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume training from checkpoint",
    )

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create config
    config = AlphaZeroConfig(
        num_selfplay_iterations=args.iterations,
        num_games_per_iteration=args.games_per_iter,
        num_simulations=args.simulations,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        cpuct=args.cpuct,
        dropout=args.dropout,
        device=args.device,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Create trainer
    trainer = AlphaZeroTrainer(config, use_tensorboard=not args.no_tensorboard)

    # Resume from checkpoint if specified
    if args.resume is not None:
        if not args.resume.exists():
            logger.error(f"Checkpoint not found: {args.resume}")
            return
        trainer.load_checkpoint(args.resume)
        # Continue training from where we left off
        remaining_iterations = args.iterations - trainer.iteration
        if remaining_iterations > 0:
            trainer.train(remaining_iterations)
        else:
            logger.info("Training already complete for specified iterations")
    else:
        trainer.train(args.iterations)


if __name__ == "__main__":
    main()
