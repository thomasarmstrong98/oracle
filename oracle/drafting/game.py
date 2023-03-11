"""Drafting Game"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np

from oracle.utils.data import NUM_HEROES


@dataclass
class DraftState:
    draft: np.ndarray
    player: int


class Game(ABC):
    @abstractmethod
    def get_string_representation(self, state: DraftState) -> str:
        pass

    @abstractmethod
    def get_valid_actions(self, state: DraftState) -> np.ndarray:
        pass

    @abstractmethod
    def take_action(self, state: DraftState, action: int) -> DraftState:
        pass

    @abstractmethod
    def get_action_size(self) -> int:
        pass

    @abstractmethod
    def get_reward(self, state: DraftState) -> float:
        pass

    @abstractmethod
    def get_start_node(self) -> DraftState:
        pass


class BasicDraft(Game):
    """Basic DOTA2 Drafting Game

    Does not include drafting schedule, picks and bans."""

    def __init__(self, reward_model: Callable[[np.ndarray], float]) -> None:
        """Initialise Game

        Args:
            reward_model (Callable[[np.ndarray], float]): Model for reward prediction from draft state.
        """
        self.reward_model = reward_model

        self.draft_array_to_string_mapping = {-1: "1", 0: "X", 1: "0"}
        self.draft_array_to_string_vect = np.vectorize(self.draft_array_to_string_mapping.get)

    def _draft_array_to_string(self, draft_array: np.ndarray) -> str:
        """Converts the np.array of draft data into a string for hashing/storing state info.

        Args:
            draft_array (np.ndarray): array of draft data

        Returns:
            str: string object for draft state storage
        """
        mapped_array = self.draft_array_to_string_vect(draft_array)

        return "".join(mapped_array)

    def get_string_representation(self, state: DraftState) -> str:
        """Convert draft state to string representation.

        Args:
            state (DraftState): Current draft state

        Returns:
            str: string representation on draft state
        """
        return self._draft_array_to_string(state.draft)

    def get_valid_actions(self, state: DraftState) -> np.ndarray:
        """Returns a binary array of lenth NUM_HEROES, with 1's representing valid hero picks

        Args:
            state (DraftState): Current state of the game.
        Returns:
            np.ndarray: Valid hero options from current game state,
            binary array where 1 represents available.
        """
        actions = np.zeros(self.get_action_size())
        actions[np.where(state.draft == 0)[0]] = 1
        return actions

    def take_action(self, state: DraftState, action: int) -> DraftState:
        """Moves current draft state to next state given action.

        E.g. pick a hero for current player and move the draft state.

        Args:
            state (DraftState): Current state
            action (int): hero_id for action

        Returns:
            DraftState: Next state
        """
        next_draft = state.draft.copy()
        next_draft[action] = +1 if not state.player else -1

        return DraftState(next_draft, int(not state.player))

    def get_action_size(self) -> int:
        """Return size of all valid and invalid actions

        Args:
            state (DraftState): Current game state

        Returns:
            int: size of all actions from this step, both valid and invalid.
        """
        return NUM_HEROES

    def is_game_terminated(self, state: DraftState) -> bool:
        """Determine if the draft is complete.

        Args:
            state (DraftState): Game state

        Returns:
            bool: Whether game is over
        """
        draft = state.draft
        return np.sum(draft == 1) == np.sum(draft == -1) == 5

    def get_model_reward(self, state: DraftState) -> float:
        """Perform model inference on game state

        Args:
            state (DraftState): Game state for inference

        Returns:
            float: win prediction
        """
        draft = state.draft
        return self.reward_model(draft.reshape(-1, 1).T)

    def get_reward(self, state: DraftState) -> float:
        """Get reward given a state, perform inference if terminated.

        Args:
            state (DraftState): Game state for inference

        Returns:
            float: Reward, 0.0 for unended games.
        """
        if self.is_game_terminated(state):
            return self.get_model_reward(state)
        else:
            return 0.0

    def get_start_node(self) -> DraftState:
        """Get root node for game.

        Returns:
            DraftState: root node
        """
        return DraftState(np.zeros(self.get_action_size()), 0)
