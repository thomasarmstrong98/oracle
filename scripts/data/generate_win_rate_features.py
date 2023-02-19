from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from oracle.utils.load import get_raw_draft_data, get_opendota_hero_embedding
from oracle.utils.data import create_draft
from oracle.win_rate.features import BasicFeatureGenerator

DEFAULT_PATH_TO_FILES = Path(__file__).parents[2] / "data" / "win_rate_dataset"


def parse_args():
    parser = ArgumentParser(
        description="Script to run mock inference of NationalPV model using data from GCP."
    )
    parser.add_argument(
        "--path_to_files",
        type=str,
        required=False,
        default=DEFAULT_PATH_TO_FILES,
    )
    args = parser.parse_args()
    return args


def main(args):
    # load in data
    dataset = get_raw_draft_data()
    hero_embedding = get_opendota_hero_embedding()
    drafts = create_draft(dataset.radiant_draft, dataset.dire_draft)

    # define the feature generator
    feature_generator = BasicFeatureGenerator(hero_embedding)

    # generate features
    feature_names, X = feature_generator(drafts)

    # save features and surrounding metadata
    np.savez(
        file=args.path_to_files,
        X=X,
        y=dataset.radiant_win,
        feature_names=feature_names,
        allow_pickle=True,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
