"""Neural network models for DOTA2 draft prediction."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeroEncoder(nn.Module):
    """Encodes individual hero picks with their embeddings."""

    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.norm(x)


class TeamEncoder(nn.Module):
    """Encodes a team of 5 heroes using attention to weight hero importance."""

    def __init__(self, hero_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hero_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hero_dim)
        self.ff = nn.Sequential(
            nn.Linear(hero_dim, hero_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hero_dim * 2, hero_dim),
        )
        self.norm2 = nn.LayerNorm(hero_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention for hero interactions within team
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x


class DraftNet(nn.Module):
    """Hero-aware draft prediction network with team composition modeling."""

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

        # Learnable hero embeddings (initialized from static embeddings)
        self.hero_embeddings = nn.Parameter(torch.zeros(num_heroes, embedding_dim))

        # Hero encoder
        self.hero_encoder = HeroEncoder(embedding_dim, hero_hidden_dim, dropout)

        # Team encoders for Radiant and Dire
        self.team_encoder = TeamEncoder(hero_hidden_dim, num_attention_heads, dropout)

        # Draft-level features
        self.draft_processor = nn.Sequential(
            nn.Linear(num_heroes, draft_hidden_dim),
            nn.LayerNorm(draft_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Combine team representations + draft-level features
        combined_dim = hero_hidden_dim * 2 + draft_hidden_dim
        self.output_head = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split input: first num_heroes dims are draft picks, rest are aggregated embeddings
        draft_picks = x[:, :self.num_heroes]  # (batch, 112) values in {-1, 0, +1}

        # Get picked hero indices for Radiant and Dire
        radiant_mask = draft_picks == 1  # (batch, 112)
        dire_mask = draft_picks == -1

        # Encode all heroes
        hero_features = self.hero_encoder(self.hero_embeddings)  # (112, hero_hidden_dim)

        # Extract team compositions
        radiant_heroes = self._extract_team_features(hero_features, radiant_mask)  # (batch, 5, dim)
        dire_heroes = self._extract_team_features(hero_features, dire_mask)

        # Encode teams with attention
        radiant_encoded = self.team_encoder(radiant_heroes)  # (batch, 5, dim)
        dire_encoded = self.team_encoder(dire_heroes)

        # Pool team representations (mean over heroes)
        radiant_rep = radiant_encoded.mean(dim=1)  # (batch, hero_hidden_dim)
        dire_rep = dire_encoded.mean(dim=1)

        # Process draft-level features
        draft_features = self.draft_processor(draft_picks)  # (batch, draft_hidden_dim)

        # Combine all representations
        combined = torch.cat([radiant_rep, dire_rep, draft_features], dim=1)

        # Final prediction
        return self.output_head(combined)

    def _extract_team_features(
        self, hero_features: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract features for picked heroes, padding to 5 heroes."""
        batch_size = mask.shape[0]
        hero_dim = hero_features.shape[1]

        # Initialize output tensor
        team_features = torch.zeros(batch_size, 5, hero_dim, device=mask.device)

        for i in range(batch_size):
            picked_indices = mask[i].nonzero(as_tuple=True)[0]
            num_picked = min(len(picked_indices), 5)
            if num_picked > 0:
                team_features[i, :num_picked] = hero_features[picked_indices[:num_picked]]

        return team_features

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(
    num_heroes: int = 112,
    embedding_dim: int = 14,
    hero_hidden_dim: int = 128,
    num_attention_heads: int = 4,
    draft_hidden_dim: int = 256,
    dropout: float = 0.2,
) -> DraftNet:
    """Create a DraftNet model."""
    return DraftNet(
        num_heroes=num_heroes,
        embedding_dim=embedding_dim,
        hero_hidden_dim=hero_hidden_dim,
        num_attention_heads=num_attention_heads,
        draft_hidden_dim=draft_hidden_dim,
        dropout=dropout,
    )
