"""Monte Carlo Tree Search"""
import math
import time
from typing import Callable, Optional, Tuple

import numpy as np

from oracle.drafting.game import DraftState, Game
from oracle.utils.data import NUM_HEROES

EPS = 1e-8


def random_policy(draft_state: DraftState) -> np.ndarray:
    """Random rollout policy

    Args:
        draft_state (DraftState): Current draft state

    Returns:
        np.ndarray: Output policy
    """
    return np.ones(NUM_HEROES) / NUM_HEROES


class ClassicMCTS:
    """Classical Monte Carlo Tree Search Algorithm"""

    def __init__(
        self,
        drafting_game: Game,
        rollout_policy: Callable[[DraftState], np.ndarray],
        search_stop_criterion: str = "time",
        search_time_threshold_seconds: int = 60 * 3,
        search_number_of_simulations: int = 1_000,
        cpuct: float = 0.1,
    ) -> None:
        """Initialise MCTS Algorithm

        Args:
            drafting_game (Game): Game to search
            rollout_policy (Callable[[DraftState], np.ndarray]): Rollout policy for simulation
            search_stop_criterion (str, optional): Whether to use time of number of simulations. Defaults to "time".
            search_time_threshold_seconds (int, optional): Defaults to 60*3.
            search_number_of_simulations (int, optional): Defaults to 1_000.
            cpuct (float, optional): UCT Exploration parameter. Defaults to 0.1.
        """
        self.search_stop_criterion = search_stop_criterion
        self.search_stop_threshold = (
            search_time_threshold_seconds
            if self.search_stop_criterion == "time"
            else search_number_of_simulations
        )

        self.game = drafting_game
        self.rollout_policy = rollout_policy
        self.cpuct = cpuct

        self.Qsa = dict()  # Q values for each (state, action)
        self.Nsa = dict()  # num. times (state, action) visited
        self.Ns = dict()  # num. times state was visited
        self.Ps = dict()  # store initial policy

        self.Es = dict()  # stores if a state is ended for states
        self.Cs = dict()  # stores children states for each state

    def run(self, start_node: Optional[DraftState] = None) -> None:
        """Run MCTS simulations from start node

        Args:
            start_node (Optional[DraftState], optional): Start node for simulations. Defaults to None.
        """
        if start_node is None:
            start_node = self.game.get_start_node()

        if self.search_stop_criterion == "time":
            timeLimit = time.time() + self.search_stop_threshold
            while time.time() < timeLimit:
                self.search(start_node)
        else:
            for _ in range(self.search_stop_threshold):
                self.search(start_node)

    def search(self, current_draft: DraftState) -> float:
        """Perform a single tree exploration and rollout from current state

        Args:
            current_draft (DraftState): State of drafting game

        Returns:
            float: Model reward
        """

        s = self.game.get_string_representation(current_draft)

        if s not in self.Es:
            self.Es[s] = self.game.get_final_reward(current_draft)

        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node of game tree, perform rollout

            # the drafting game is a strict tree, rather than a DAG
            # this means once we get to an unvisited leaf, all
            # following states are also unvisited and thus our
            # flag for performing rollout or UCT is just self.Ps

            # get rollout policy from state
            self.Ps[s] = self.rollout_policy(current_draft)
            valid_actions = self.game.get_valid_actions(current_draft)
            self.Ps[s] = self.Ps[s] * valid_actions
            norm = np.sum(self.Ps[s])
            self.Ps[s] /= norm

            self.Cs[s] = valid_actions
            self.Ns[s] = 0  # leaf node has been visited

            action = np.random.choice(len(self.Ps[s]), 1, p=self.Ps[s])

            next_state = self.game.take_action(current_draft, action)
            return -self.search(next_state)

        # seen the state before therefore perfom UCT for child
        valid_actions = self.Cs[s]
        current_best = -float("inf")
        best_action = None

        for a in range(self.game.get_action_size()):
            if valid_actions[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                        1 + self.Nsa[(s, a)]
                    )

                else:
                    u = self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > current_best:
                    current_best = u
                    best_action = a

        a = best_action
        next_state = self.game.take_action(current_draft, a)
        v = self.search(next_state)

        # backpropagate the game reward up the tree
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v

    def forward(self, state: DraftState) -> Tuple[DraftState, int]:
        """Find the best action without exploration from the state.

        Args:
            state (DraftState): State to find the best action from.

        Returns:
            Tuple[DraftState, int]: Next draft state and the hero/action picked
        """
        s = self.game.get_string_representation(state)

        current_best = -float("inf")
        best_action = None
        valid_actions = self.Cs[s]

        for a in range(self.game.get_action_size()):
            if valid_actions[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)]
                else:
                    u = 0

                if u > current_best:
                    current_best = u
                    best_action = a

        assert best_action is not None

        return self.game.take_action(state, best_action), best_action