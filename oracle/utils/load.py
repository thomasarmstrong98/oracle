import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from oracle.utils.data import DATA_DIRECTORY, DOTA_HERO_IDS_IN_DATASET
from oracle.utils.logger import getLogger

logger = getLogger(__name__)

OPENDOTA_HERO_STATS_QUERY = "https://api.opendota.com/api/heroStats?api_key="
DEFAULT_PATH_TO_EMBEDDING_DATA = Path(DATA_DIRECTORY / "hero_embedding.pickle")


def get_opendota_hero_embedding(
    path_to_local: Optional[Path] = DEFAULT_PATH_TO_EMBEDDING_DATA,
    reset_ids: bool = True,
    use_local: bool = True,
) -> pd.DataFrame:
    if not use_local:
        logger.warning(
            "Since project creation, certain embedding features have been deleted/changed. "
            "Highly recommended to use provided data."
        )
        resp = requests.get(OPENDOTA_HERO_STATS_QUERY)
        assert resp.ok
        hero_stats = pd.DataFrame(json.loads(resp.content)).set_index("id")

        # get roles embedding for each hero
        roles_encoding = hero_stats["roles"].str.join("|").str.get_dummies()
        embedding = pd.concat(
            [
                pd.get_dummies(hero_stats[["attack_type", "primary_attr"]]),
                roles_encoding,
            ],
            axis=1,
        )
    else:
        embedding = pd.read_pickle(path_to_local)

    embedding = (
        embedding.loc[DOTA_HERO_IDS_IN_DATASET].reset_index(drop=True) if reset_ids else embedding
    )
    return embedding


def get_opendota_hero_stats(
    path_to_local: Optional[Path] = None, reset_ids: bool = True
) -> pd.DataFrame:
    if path_to_local is None:
        resp = requests.get(OPENDOTA_HERO_STATS_QUERY)
        assert resp.ok
        hero_stats = pd.DataFrame(json.loads(resp.content)).set_index("id")
    else:
        hero_stats = pd.read_pickle(path_to_local)

    hero_stats = (
        hero_stats.loc[DOTA_HERO_IDS_IN_DATASET].reset_index(drop=True) if reset_ids else hero_stats
    )
    return hero_stats


@dataclass
class RawDraftData:
    """Object for storing the raw DOTA draft data.

    RawDraftData contains data for around 2M dota matches.
    It contains the match outcome (radiant_win) for each match,
    in addition to the drafts for both teams.

    Note:
    In the dataset, the hero_id set goes up to 114, however there are only 112
    heroes in the dataset. This is a known issue/feature, id 24 is permanently
    blank/unassigned and id 108 was preassigned to a hero that was released
    after the datacollection.


    """

    radiant_win: np.ndarray
    radiant_draft: np.ndarray
    dire_draft: np.ndarray
    last_hero_id: int


def get_raw_draft_data():
    data_dir = Path(__file__).parents[2]
    try:
        radiant_win, radiant_draft, dire_draft, _, _, _, num_heroes = pd.read_pickle(
            data_dir / "data" / "dota.pickle"
        )

        # filter out games with hero id == 0.
        games = np.concatenate((radiant_draft, dire_draft), axis=1)
        drop_games_with_zero_id = ~(games == 0).any(axis=1)
        radiant_draft = radiant_draft[drop_games_with_zero_id]
        dire_draft = dire_draft[drop_games_with_zero_id]
        radiant_win = radiant_win[drop_games_with_zero_id]

        return RawDraftData(radiant_win, radiant_draft, dire_draft, num_heroes)
    except Exception:
        raise Exception(
            "Could not find draft data locally, "
            "download from: https://www.dropbox.com/s/vy4zei33725l8a4/dota.pickle?dl=0"
        )
