from pathlib import Path

import numpy as np

import oracle

DATA_DIRECTORY = Path(oracle.__file__).parents[1] / "data"
CONFIG_DIRECTORY = Path(oracle.__file__).parents[1] / "configs"

#  skip 24 and 108
DOTA_HERO_IDS_IN_DATASET = list(range(1, 24)) + list(range(25, 108)) + list(range(109, 115))
DOTA_HERO_ID_TO_INDEX_MAPPING = {val: idx for idx, val in enumerate(DOTA_HERO_IDS_IN_DATASET)}
DOTA_HERO_INDEX_TO_ID_MAPPING = {idx: _id for _id, idx in DOTA_HERO_ID_TO_INDEX_MAPPING.items()}
NUM_HEROES = len(DOTA_HERO_IDS_IN_DATASET)


def create_draft(
    radiant_drafts: np.ndarray, dire_drafts: np.ndarray, map_ids: bool = True
) -> np.ndarray:
    if map_ids:
        # concatenate and map hero_ids to numeric true ids
        games = np.vectorize(DOTA_HERO_ID_TO_INDEX_MAPPING.get)(
            np.concatenate((radiant_drafts, dire_drafts), axis=1)
        )
    else:
        games = np.concatenate((radiant_drafts, dire_drafts), axis=1)
    # create the flattened/encoded draft TODO - make this quicker, takes 4sec on 3M rows
    drafts = np.empty((games.shape[0], NUM_HEROES), dtype=np.int8)
    for game, draft in enumerate(games):
        drafts[game][draft[:5]] = +1
        drafts[game][draft[5:]] = -1

    return drafts
