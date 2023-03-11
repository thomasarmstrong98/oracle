from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from oracle.drafting.game import BasicDraft
from oracle.drafting.mcts import AlphaZeroMCTS


@dataclass
class SelfPlayExample:
    player: int
    policy: np.ndarray
    outcome: Optional[float] = 0.0


class Coach:
    def __init__(self, draft_game: BasicDraft, a0_nnet, num_rounds_of_self_play: int) -> None:
        self.draft_game = draft_game
        self.a0_nnet = a0_nnet
        self.mcts = AlphaZeroMCTS(draft_game, a0_nnet)
        self.num_rounds_of_self_play = num_rounds_of_self_play

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
            pi = self.mcts.run(draft_state, return_action_policy=True)
            assert pi is not None
            train_examples.append(SelfPlayExample(draft_state.player, pi, None))
            action = np.random.choice(len(pi), p=pi)
            draft_state = self.draft_game.take_action(draft_state, action)

            r = self.draft_game.get_reward(draft_state)

            if not np.isclose(r, 0.0, rtol=1e-6):
                # end of the game, update the path with corresponding reward.
                return [
                    SelfPlayExample(
                        sample.player,
                        sample.policy,
                        r * ((-1) ** (sample.player != draft_state.player)),
                    )
                    for sample in train_examples
                ]

    def learn(self):
        training_examples = list()

        for _ in tqdm(range(self.num_rounds_of_self_play), desc="Self play iterations."):
            self.mcts = AlphaZeroMCTS(self.draft_game, self.a0_nnet)
            training_examples += self.selfplay_round()
