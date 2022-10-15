from copy import deepcopy
from typing import Callable, Iterable
import joblib
import numpy as np
from numpy.typing import ArrayLike

TOTAL_HERO_NUM = 111
WIN_RATE_MODEL = joblib.load("../win_rate/wr_basic_logisitic_regression.p")


def load_reward_function() -> Callable[[ArrayLike], float]:
    model = joblib.load("../win_rate/wr_basic_logisitic_regression.p")
    embedding = np.load("../win_rate/basic_hero_embedding.npy")

    return lambda x: model.predict_proba(
        np.concatenate((x @ embedding, x)).reshape(1, -1)
    )[0]


reward_function = load_reward_function()


class Action:
    def __init__(self, player, hero):
        self.player: int = player
        self.hero: int = hero

    def __str__(self):
        return str(self.hero)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__
            and self.hero == other.hero
            and self.player == other.player
        )

    def __hash__(self):
        return hash((self.hero, self.player))


class BasicDotaDraftingState:
    def __init__(self):
        self.draft: ArrayLike = np.zeros((TOTAL_HERO_NUM))
        self.current_player: int = 1

    def get_draft(self) -> ArrayLike:
        return self.draft

    def get_current_player(self) -> int:
        return self.current_player

    def get_unpicked_heroes(self) -> Iterable:
        return np.argwhere(self.draft == 0).flatten()

    def get_possible_actions(self) -> Iterable[Action]:
        possible_actions = [
            Action(player=self.current_player, hero=unpicked_hero)
            for unpicked_hero in self.get_unpicked_heroes()
        ]
        return possible_actions

    def take_action(self, action: Action):
        """
        Current player choses a hero
        """

        new_state = deepcopy(self)
        new_state.draft[action.hero] = action.player
        new_state.current_player = self.current_player * -1
        return new_state

    def is_terminal(self) -> bool:
        """
        Draft ends when both teams (+1/-1) have picked 5 heroes.
        """
        return np.sum(self.draft == 1) == np.sum(self.draft == -1) == 5

    def get_reward(self) -> float:
        """
        Predict the probabilty of the selected draft winning for the
        current player.
        """

        return reward_function(self.draft)[int(0.5 * (1 - self.current_player))]
