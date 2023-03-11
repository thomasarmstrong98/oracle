from typing import Tuple, TypeAlias

import numpy as np
import pandas as pd

FeatureNames: TypeAlias = np.ndarray
Features: TypeAlias = np.ndarray


class BasicFeatureGenerator:
    def __init__(self, hero_embeddings: pd.DataFrame) -> None:
        self.hero_embeddings = hero_embeddings
        self.hero_embedding_matrix = self.hero_embeddings.values.astype(np.int0)

    def __call__(self, drafts: np.ndarray) -> Tuple[FeatureNames, Features]:
        features = np.concatenate((drafts @ self.hero_embedding_matrix, drafts), axis=1)
        names = np.concatenate(
            (
                self.hero_embeddings.columns.values,
                np.asarray([f"hero_{_id}" for _id in range(drafts.shape[1])]),
            )
        )
        return names, features
