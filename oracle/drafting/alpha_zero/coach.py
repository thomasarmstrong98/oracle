from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from oracle.drafting.game import BasicDraft
from oracle.drafting.mcts import AggregatorClassicMCTS, BaseMCTS


@dataclass
class SelfPlayExample:
    player: int
    draft: np.ndarray
    policy: np.ndarray
    outcome: Optional[float] = 0.0


class Coach:
    def __init__(
        self,
        draft_game: BasicDraft,
        mcts: Union[AggregatorClassicMCTS, BaseMCTS],
        tree_reset_interval: int = 10,
    ) -> None:
        self.draft_game = draft_game
        self.mcts = mcts
        self.tree_reset_interval = tree_reset_interval
        self.tree_round_count = 0

    def selfplay_round(self) -> List[SelfPlayExample]:
        """A single round of self-play.

        Returns:
            List[SelfPlayExample]: List of examples for model to train with.
        """
        # start with a new game
        draft_state = self.draft_game.get_start_node()
        train_examples = list()

        while True:
            # until the game is finished
            if self.tree_round_count == self.tree_reset_interval:
                print("Tree has been reset!")
                self.mcts.reset()
            pi = self.mcts.run(draft_state, return_action_policy=True)
            assert pi is not None
            train_examples.append(SelfPlayExample(draft_state.player, draft_state.draft, pi, None))
            action = np.random.choice(len(pi), p=pi)
            draft_state = self.draft_game.take_action(draft_state, action)

            if self.draft_game.is_game_terminated(draft_state):
                r = self.draft_game.get_reward(draft_state)

                # end of the game, update the path with corresponding reward.
                return [
                    SelfPlayExample(
                        sample.player,
                        sample.draft,
                        sample.policy,
                        r * ((-1) ** (sample.player != draft_state.player)),
                    )
                    for sample in train_examples
                ]

            self.tree_round_count += 1

    @classmethod
    def convert_examples_to_numpy(
        cls, examples: List[SelfPlayExample]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Converts a list of self-play examples to numpy arrays.

        Args:
            examples (List[SelfPlayExample]): output of self play

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: numpy array version of selfplay examples
        """
        players = np.asarray([example.player for example in examples])
        drafts = np.asarray([example.draft for example in examples])
        policies = np.asarray([example.policy for example in examples])
        outcomes = np.asarray([example.outcome for example in examples])

        return players, drafts, policies, outcomes
