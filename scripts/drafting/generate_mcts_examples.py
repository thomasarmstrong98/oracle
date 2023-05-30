"""Generate examples from ClassicMCTS algorithm, for use in training AlphaZero model
"""
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

import numpy as np

from oracle.drafting.alpha_zero.coach import Coach
from oracle.drafting.game import BasicDraft
from oracle.drafting.mcts import AggregatorClassicMCTS, random_policy
from oracle.utils.data import DATA_DIRECTORY
from oracle.utils.load import get_opendota_hero_embedding
from oracle.win_rate.features import BasicFeatureGenerator
from oracle.win_rate.models.base import (
    WinRateClassificationWrapper,
    WinRateModel,
    WinRateModelConfig,
)


def parse_args():
    parser = ArgumentParser(description="Script to generate training examples using classic MCTS")
    parser.add_argument(
        "--path_for_saved_examples",
        type=Path,
        required=False,
        default=DATA_DIRECTORY / "training_examples",
    )
    parser.add_argument("--number_of_self_play_rounds", type=int, required=False, default=50)
    parser.add_argument("--tree_reset_interval", type=int, required=False, default=5)
    args = parser.parse_args()
    return args


def default_load_win_rate_model() -> WinRateClassificationWrapper:
    # win rate model
    model_config = WinRateModelConfig(use_gpu=False, transform_prediction_for_reward=True)
    base_model = WinRateModel(model_config)
    base_model.load_model()

    # feature generator for model
    feature_generator = BasicFeatureGenerator(get_opendota_hero_embedding())

    # pipeline wrapper
    wr_model = WinRateClassificationWrapper(
        feature_generator=feature_generator, win_rate_model=base_model
    )
    return wr_model


def save_examples_locally(
    examples: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], file_path: Path
) -> None:
    players, drafts, policies, outcomes = examples
    np.savez(players=players, drafts=drafts, policies=policies, outcomes=outcomes, file=file_path)


def main(
    number_of_self_play_rounds: int = 50,
    tree_reset_interval: int = 10,
    number_of_parrallel_trees: int = 8,
    number_of_processes: int = 10,
    time_limit_per_tree_seconds: int = 15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    examples = list()

    wr_model = default_load_win_rate_model()
    game = BasicDraft(wr_model)
    mcts = AggregatorClassicMCTS(
        drafting_game=game,
        parrallel_trees=number_of_parrallel_trees,
        number_processes=number_of_processes,
        rollout_policy=random_policy,
        search_stop_criterion="time",
        search_time_threshold_seconds=time_limit_per_tree_seconds,
    )

    coach = Coach(game, mcts, tree_reset_interval=tree_reset_interval)

    for _ in range(number_of_self_play_rounds):
        examples += coach.selfplay_round()

    examples = coach.convert_examples_to_numpy(examples)

    return examples


if __name__ == "__main__":
    args = parse_args()
    _start = time.time()
    examples = main(
        number_of_self_play_rounds=args.number_of_self_play_rounds,
        tree_reset_interval=args.tree_reset_interval,
    )
    _end = time.time()
    print(f"Took: {_end - _start} for {args.number_of_self_play_rounds} self-play rounds.")
    save_examples_locally(examples, args.path_for_saved_examples)
