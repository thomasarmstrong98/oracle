"""Data loading and processing utilities."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

import oracle

OPENDOTA_HERO_STATS_QUERY = "https://api.opendota.com/api/heroStats?api_key="

DATA_DIRECTORY = Path(oracle.__file__).parents[1] / "data"
CONFIG_DIRECTORY = Path(oracle.__file__).parents[1] / "configs"
DEFAULT_PATH_TO_EMBEDDING_DATA = Path(DATA_DIRECTORY / "hero_embedding.pickle")

# Skip heroes 24 and 108 (missing from dataset)
DOTA_HERO_IDS_IN_DATASET = list(range(1, 24)) + list(range(25, 108)) + list(range(109, 115))
DOTA_HERO_ID_TO_INDEX_MAPPING = {val: idx for idx, val in enumerate(DOTA_HERO_IDS_IN_DATASET)}
DOTA_HERO_INDEX_TO_ID_MAPPING = {idx: _id for _id, idx in DOTA_HERO_ID_TO_INDEX_MAPPING.items()}
DOTA_HERO_INDICIES = list(DOTA_HERO_ID_TO_INDEX_MAPPING.values())
NUM_HEROES = len(DOTA_HERO_INDICIES)


@dataclass
class DraftData:
    """Container for DOTA draft match data."""
    radiant_win: np.ndarray
    radiant_draft: np.ndarray
    dire_draft: np.ndarray
    last_hero_id: int


def get_opendota_hero_embedding(
    path_to_local: Optional[Path] = DEFAULT_PATH_TO_EMBEDDING_DATA,
    reset_ids: bool = True,
    use_local: bool = True,
) -> pd.DataFrame:
    """Load hero embeddings from local file or OpenDota API."""
    if not use_local and path_to_local is not None:
        resp = requests.get(OPENDOTA_HERO_STATS_QUERY)
        assert resp.ok
        hero_stats = pd.DataFrame(json.loads(resp.content)).set_index("id")
        roles_encoding = hero_stats["roles"].str.join("|").str.get_dummies()
        embedding = pd.concat([
            pd.get_dummies(hero_stats[["attack_type", "primary_attr"]]),
            roles_encoding,
        ], axis=1)
    else:
        embedding = pd.read_pickle(path_to_local)

    if reset_ids:
        embedding = embedding.loc[DOTA_HERO_IDS_IN_DATASET].reset_index(drop=True)

    return embedding


def get_opendota_hero_stats(
    path_to_local: Optional[Path] = None, reset_ids: bool = True
) -> pd.DataFrame:
    """Load hero stats from local file or OpenDota API."""
    if path_to_local is None:
        resp = requests.get(OPENDOTA_HERO_STATS_QUERY)
        assert resp.ok
        hero_stats = pd.DataFrame(json.loads(resp.content)).set_index("id")
    else:
        hero_stats = pd.read_pickle(path_to_local)

    if reset_ids:
        hero_stats = hero_stats.loc[DOTA_HERO_IDS_IN_DATASET].reset_index(drop=True)

    return hero_stats


def get_hero_names(use_local: bool = True) -> dict:
    """Get mapping of hero indices to localized names.

    Args:
        use_local: Whether to use local cache or fetch from API

    Returns:
        Dictionary mapping hero index (0-111) to hero name
    """
    try:
        hero_stats = get_opendota_hero_stats(reset_ids=True)
        if 'localized_name' in hero_stats.columns:
            return {idx: name for idx, name in enumerate(hero_stats['localized_name'])}
        elif 'name' in hero_stats.columns:
            # Strip "npc_dota_hero_" prefix if present
            names = hero_stats['name'].str.replace('npc_dota_hero_', '', regex=False)
            return {idx: name for idx, name in enumerate(names)}
    except Exception:
        pass

    # Fallback: return generic names
    return {i: f"Hero_{i}" for i in range(NUM_HEROES)}


def read_draft_data(path: Path) -> DraftData:
    """Load draft data from pickle file."""
    try:
        radiant_win, radiant_draft, dire_draft, _, _, _, num_heroes = pd.read_pickle(path)

        # Filter out games with hero_id == 0
        games = np.concatenate((radiant_draft, dire_draft), axis=1)
        valid_games = ~(games == 0).any(axis=1)

        return DraftData(
            radiant_win=radiant_win[valid_games],
            radiant_draft=radiant_draft[valid_games],
            dire_draft=dire_draft[valid_games],
            last_hero_id=num_heroes
        )
    except Exception:
        raise Exception(
            "Could not find draft data. Download from: "
            "https://www.dropbox.com/s/vy4zei33725l8a4/dota.pickle?dl=0"
        )


def create_draft(
    radiant_drafts: np.ndarray, dire_drafts: np.ndarray, map_ids: bool = True
) -> np.ndarray:
    """Convert draft picks to binary encoding (+1 for radiant, -1 for dire)."""
    if map_ids:
        games = np.vectorize(DOTA_HERO_ID_TO_INDEX_MAPPING.get)(
            np.concatenate((radiant_drafts, dire_drafts), axis=1)
        )
    else:
        games = np.concatenate((radiant_drafts, dire_drafts), axis=1)

    drafts = np.zeros((games.shape[0], NUM_HEROES), dtype=np.int8)
    for i, draft in enumerate(games):
        drafts[i][draft[:5]] = 1   # Radiant picks
        drafts[i][draft[5:]] = -1  # Dire picks

    return drafts
