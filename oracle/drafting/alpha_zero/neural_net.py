"""AlphaZero neural network for DOTA2 draft policy and value prediction."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from oracle.models import HeroEncoder, TeamEncoder


class AlphaZeroNet(nn.Module):
    """Dual-head network for policy (hero selection) and value (position eval)."""

    def __init__(
        self,
        num_heroes: int = 112,
        embedding_dim: int = 14,
        hero_hidden_dim: int = 128,
        num_attention_heads: int = 4,
        draft_hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_heroes = num_heroes
        self.embedding_dim = embedding_dim

        # Learnable hero embeddings
        self.hero_embeddings = nn.Parameter(torch.zeros(num_heroes, embedding_dim))

        # Shared backbone - encode heroes and team compositions
        self.hero_encoder = HeroEncoder(embedding_dim, hero_hidden_dim, dropout)
        self.team_encoder = TeamEncoder(hero_hidden_dim, num_attention_heads, dropout)

        # Draft-level features
        self.draft_processor = nn.Sequential(
            nn.Linear(num_heroes, draft_hidden_dim),
            nn.LayerNorm(draft_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Shared representation layer
        combined_dim = hero_hidden_dim * 2 + draft_hidden_dim
        self.shared_head = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Policy head - outputs log probabilities over heroes
        self.policy_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_heroes),
        )

        # Value head - outputs position evaluation
        self.value_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh(),  # Output in range [-1, 1]
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def initialize_embeddings(self, embeddings: torch.Tensor):
        """Initialize hero embeddings from pre-trained embeddings."""
        with torch.no_grad():
            self.hero_embeddings.copy_(embeddings)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning policy logits and value.

        Args:
            x: Draft state tensor (batch, num_heroes) with values in {-1, 0, +1}

        Returns:
            policy_logits: (batch, num_heroes) raw logits for hero selection
            value: (batch, 1) position evaluation in [-1, 1]
        """
        draft_picks = x[:, : self.num_heroes]  # (batch, 112)

        # Get picked hero indices for Radiant and Dire
        radiant_mask = draft_picks == 1
        dire_mask = draft_picks == -1

        # Encode all heroes
        hero_features = self.hero_encoder(self.hero_embeddings)  # (112, hero_hidden_dim)

        # Extract team compositions
        radiant_heroes = self._extract_team_features(hero_features, radiant_mask)
        dire_heroes = self._extract_team_features(hero_features, dire_mask)

        # Encode teams with attention
        radiant_encoded = self.team_encoder(radiant_heroes)
        dire_encoded = self.team_encoder(dire_heroes)

        # Pool team representations
        radiant_rep = radiant_encoded.mean(dim=1)  # (batch, hero_hidden_dim)
        dire_rep = dire_encoded.mean(dim=1)

        # Process draft-level features
        draft_features = self.draft_processor(draft_picks)

        # Combine representations
        combined = torch.cat([radiant_rep, dire_rep, draft_features], dim=1)

        # Shared processing
        shared_features = self.shared_head(combined)

        # Dual heads
        policy_logits = self.policy_head(shared_features)  # (batch, num_heroes)
        value = self.value_head(shared_features)  # (batch, 1)

        return policy_logits, value

    def _extract_team_features(
        self, hero_features: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract features for picked heroes, padding to 5 heroes."""
        batch_size = mask.shape[0]
        hero_dim = hero_features.shape[1]

        team_features = torch.zeros(batch_size, 5, hero_dim, device=mask.device)

        for i in range(batch_size):
            picked_indices = mask[i].nonzero(as_tuple=True)[0]
            num_picked = min(len(picked_indices), 5)
            if num_picked > 0:
                team_features[i, :num_picked] = hero_features[picked_indices[:num_picked]]

        return team_features

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class NeuralNetWrapper:
    """Wrapper for AlphaZeroNet to interface with MCTS."""

    def __init__(
        self,
        model: AlphaZeroNet,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, state) -> Tuple[np.ndarray, float]:
        """Predict policy and value for a single draft state.

        Args:
            state: Draft state - either DraftState object or numpy array (num_heroes,)

        Returns:
            policy: Probability distribution over heroes (num_heroes,)
            value: Position evaluation scalar in [-1, 1]
        """
        # Handle both DraftState objects and numpy arrays
        from oracle.drafting.game import DraftState

        if isinstance(state, DraftState):
            draft = state.draft
        else:
            draft = state

        # Convert to tensor and add batch dimension
        draft_tensor = torch.from_numpy(draft).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, value = self.model(draft_tensor)

            # Convert policy logits to probabilities
            policy = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
            value = value.item()

        return policy, value

    def predict_batch(self, drafts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict policy and value for a batch of draft states.

        Args:
            drafts: Draft arrays (batch, num_heroes)

        Returns:
            policies: Probability distributions (batch, num_heroes)
            values: Position evaluations (batch,)
        """
        drafts_tensor = torch.from_numpy(drafts).float().to(self.device)

        with torch.no_grad():
            policy_logits, values = self.model(drafts_tensor)
            policies = F.softmax(policy_logits, dim=1).cpu().numpy()
            values = values.squeeze().cpu().numpy()

        return policies, values

    def train_on_batch(
        self,
        drafts: np.ndarray,
        target_policies: np.ndarray,
        target_values: np.ndarray,
        optimizer: torch.optim.Optimizer,
        policy_weight: float = 1.0,
        value_weight: float = 1.0,
    ) -> Tuple[float, float, float]:
        """Train model on a batch of self-play examples.

        Args:
            drafts: Draft states (batch, num_heroes)
            target_policies: Target policy distributions (batch, num_heroes)
            target_values: Target position evaluations (batch,)
            optimizer: PyTorch optimizer
            policy_weight: Weight for policy loss
            value_weight: Weight for value loss

        Returns:
            total_loss: Combined loss
            policy_loss: Policy component
            value_loss: Value component
        """
        self.model.train()

        # Convert to tensors
        drafts_tensor = torch.from_numpy(drafts).float().to(self.device)
        target_policies_tensor = torch.from_numpy(target_policies).float().to(self.device)
        target_values_tensor = torch.from_numpy(target_values).float().to(self.device)

        # Forward pass
        policy_logits, values = self.model(drafts_tensor)

        # Policy loss (cross-entropy between distributions)
        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = -(target_policies_tensor * log_probs).sum(dim=1).mean()

        # Value loss (MSE)
        value_loss = F.mse_loss(values.squeeze(), target_values_tensor)

        # Combined loss
        total_loss = policy_weight * policy_loss + value_weight * value_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        self.model.eval()

        return (
            total_loss.item(),
            policy_loss.item(),
            value_loss.item(),
        )

    def save_checkpoint(self, path: Path, optimizer: Optional[torch.optim.Optimizer] = None):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(
        self, path: Path, optimizer: Optional[torch.optim.Optimizer] = None
    ):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def create_alphazero_net(
    num_heroes: int = 112,
    embedding_dim: int = 14,
    hero_hidden_dim: int = 128,
    num_attention_heads: int = 4,
    draft_hidden_dim: int = 256,
    dropout: float = 0.2,
) -> AlphaZeroNet:
    """Create an AlphaZero network."""
    return AlphaZeroNet(
        num_heroes=num_heroes,
        embedding_dim=embedding_dim,
        hero_hidden_dim=hero_hidden_dim,
        num_attention_heads=num_attention_heads,
        draft_hidden_dim=draft_hidden_dim,
        dropout=dropout,
    )
