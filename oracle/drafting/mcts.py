"""Monte Carlo Tree Search implementations."""

import math
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Dict, Optional, Tuple

import numpy as np

from oracle.drafting.game import DraftState, Game

EPS = 1e-8


def random_policy(game: Game, state: DraftState) -> np.ndarray:
    """Uniform random policy over valid actions."""
    uniform = np.ones(game.get_action_size()) / game.get_action_size()
    valid_actions = game.get_valid_actions(state)
    uniform = uniform * valid_actions
    return uniform / np.sum(uniform)


class BaseMCTS(ABC):
    def __init__(
        self,
        drafting_game: Game,
        search_stop_criterion: str = "time",
        search_time_threshold_seconds: int = 60 * 3,
        search_number_of_simulations: int = 1_000,
        cpuct: float = 0.1,
    ) -> None:
        self.search_stop_criterion = search_stop_criterion
        self.search_stop_threshold = (
            search_time_threshold_seconds
            if self.search_stop_criterion == "time"
            else search_number_of_simulations
        )
        self.game = drafting_game
        self.cpuct = cpuct

        # Tree search state
        self.Qsa: dict[tuple[str, int], float] = {}
        self.Nsa: dict[tuple[str, int], int] = {}
        self.Ns: dict[str, int] = {}
        self.Ps: dict[str, np.ndarray] = {}
        self.Es: dict[str, float] = {}
        self.Cs: dict[str, np.ndarray] = {}

    def run(
        self, start_node: Optional[DraftState] = None, return_action_policy: Optional[bool] = True
    ) -> Optional[np.ndarray]:
        """Run MCTS simulations from start node

        Args:
            start_node (Optional[DraftState], optional): Start node for simulations. Defaults to None.
            return_action_policy (Optional[bool], optional): If to return probability move vector. Defaults to True.

        Returns:
            Optional[np.ndarray]: Probability move vector if return_action_policy
        """
        if start_node is None:
            start_node = self.game.get_start_node()

        if self.search_stop_criterion == "time":
            count = 0
            timeLimit = time.time() + self.search_stop_threshold
            while time.time() < timeLimit:
                self.search(start_node)
                count += 1
        else:
            for _ in range(self.search_stop_threshold):
                self.search(start_node)

        if return_action_policy:
            return self.get_action_policy(start_node)

    @abstractmethod
    def search(self, current_draft: DraftState) -> float:
        """Perform a single tree exploration.

        Args:
            current_draft (DraftState): State of drafting game

        Returns:
            float: Model reward
        """
        return 0.0

    def get_action_policy(self, state: DraftState) -> np.ndarray:
        """Returns a action probability vector from current state

        Args:
            state (DraftState): start node

        Returns:
            np.ndarray: vector of action probabilities.
        """
        s = self.game.get_string_representation(state)
        counts = np.asarray(
            [
                self.Nsa[(s, a)] if (s, a) in self.Nsa else 0
                for a in range(self.game.get_action_size())
            ]
        )
        return counts / np.sum(counts)

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

    def reset(self) -> None:
        """Hard reset the data stored in the tree"""

        self.__init__(
            drafting_game=self.game,
            search_stop_criterion=self.search_stop_criterion,
            search_time_threshold_seconds=self.search_stop_threshold,
            search_number_of_simulations=self.search_stop_threshold,
            cpuct=self.cpuct,
        )

    def set_data(
        self,
        Qsa: Dict[Tuple[str, int], float],
        Nsa: Dict[Tuple[str, int], int],
        Ns: Dict[str, int],
        Ps: Dict[str, np.ndarray],
        Es: Dict[str, float],
        Cs: Dict[str, np.ndarray],
    ):
        """Set the graph exploration information explicity - used by aggregator MCTS class

        Args:
            Qsa (Dict[str, float]): (state, action) values
            Nsa (Dict[str, int]): (state, action) counts
            Ns (Dict[str, int]): state counts
            Ps (Dict[str, np.ndarray]): state policy
            Es (Dict[str, bool]): state terminal state info
            Cs (Dict[str, np.ndarray]): children information
        """
        self.Qsa = Qsa  # Q values for each (state, action)
        self.Nsa = Nsa  # num. times (state, action) visited
        self.Ns = Ns  # num. times state was visited
        self.Ps = Ps  # store initial policy

        self.Es = Es  # stores if a state is ended for states
        self.Cs = Cs


class ClassicMCTS(BaseMCTS):
    """Classical Monte Carlo Tree Search Algorithm"""

    def __init__(
        self,
        drafting_game: Game,
        rollout_policy: Callable[[Game, DraftState], np.ndarray],
        search_stop_criterion: str = "time",
        search_time_threshold_seconds: int = 60 * 5,
        search_number_of_simulations: int = 1_000,
        cpuct: float = 0.1,
    ) -> None:
        """Initialise Classic MCTS Algorithm

        Args:
            drafting_game (Game): Game to search
            rollout_policy (Callable[[DraftState], np.ndarray]): Rollout policy for simulation
            search_stop_criterion (str, optional): Whether to use time of number of simulations. Defaults to "time".
            search_time_threshold_seconds (int, optional): Defaults to 60*3.
            search_number_of_simulations (int, optional): Defaults to 1_000.
            cpuct (float, optional): UCT Exploration parameter. Defaults to 0.1.
        """
        super().__init__(
            drafting_game,
            search_stop_criterion,
            search_time_threshold_seconds,
            search_number_of_simulations,
            cpuct,
        )

        self.rollout_policy = rollout_policy

    def search(self, current_draft: DraftState) -> float:
        """Perform a single tree exploration and rollout from current state

        Args:
            current_draft (DraftState): State of drafting game

        Returns:
            float: Game reward
        """

        s = self.game.get_string_representation(current_draft)

        if self.game.is_game_terminated(current_draft):
            self.Es[s] = self.game.get_reward(current_draft)
            return -self.Es[s]
        else:
            self.Es[s] = 0.0

        if s not in self.Ps:
            # leaf node of game tree, perform rollout

            # the drafting game is a strict tree, rather than a cyclic graph
            # this means once we get to an unvisited leaf, all
            # following states are also unvisited and thus our
            # flag for performing rollout or UCT is just self.Ps

            # get rollout policy from state
            self.Ps[s] = self.rollout_policy(self.game, current_draft)
            self.Cs[s] = self.game.get_valid_actions(current_draft)
            self.Ns[s] = 0  # leaf node has been visited

            action = np.random.choice(len(self.Ps[s]), p=self.Ps[s])

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
                    u = 1.0 + self.cpuct * self.Ps[s][a] * math.sqrt(
                        self.Ns[s]
                    )  # Over optimistic start
                if u > current_best:
                    current_best = u
                    best_action = a

        a = best_action
        assert a is not None
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


class AlphaZeroMCTS(BaseMCTS):
    def __init__(
        self,
        drafting_game: Game,
        game_net: Callable[[DraftState], Tuple[np.ndarray, float]],
        search_stop_criterion: str = "time",
        search_time_threshold_seconds: int = 60 * 1,
        search_number_of_simulations: int = 1000,
        cpuct: float = 0.1,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        temperature: float = 1.0,
    ) -> None:
        """Initialise AlphaZero MCTS Algorithm

        Args:
            drafting_game (Game): Game to search
            game_net (Callable[[DraftState], Tuple[np.ndarray, float]]): Neural Net for policy and value estimation.
            search_stop_criterion (str, optional): Whether to use time of number of simulations. Defaults to "time".
            search_time_threshold_seconds (int, optional): Defaults to 60*3.
            search_number_of_simulations (int, optional): Defaults to 1_000.
            cpuct (float, optional): UCT Exploration parameter. Defaults to 0.1.
            dirichlet_alpha (float, optional): Dirichlet noise parameter for exploration. Defaults to 0.3.
            dirichlet_epsilon (float, optional): Weight of Dirichlet noise at root. Defaults to 0.25.
            temperature (float, optional): Temperature for action selection. Defaults to 1.0.
        """
        super().__init__(
            drafting_game,
            search_stop_criterion,
            search_time_threshold_seconds,
            search_number_of_simulations,
            cpuct,
        )

        self.game_net = game_net
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.temperature = temperature
        self.root_state = None  # Track root for Dirichlet noise

    def search(self, current_draft: DraftState) -> float:
        """Perform a single tree exploration with neural net policy/value from current state.

        Args:
            current_draft (DraftState): State of drafting game

        Returns:
            float: Game reward
        """

        s = self.game.get_string_representation(current_draft)

        if s not in self.Es:
            self.Es[s] = self.game.get_reward(current_draft)

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
            self.Ps[s], v = self.game_net(current_draft)
            valid_actions = self.game.get_valid_actions(current_draft)
            self.Ps[s] = self.Ps[s] * valid_actions
            norm = np.sum(self.Ps[s])
            self.Ps[s] /= norm

            # Apply Dirichlet noise at root for exploration (AlphaZero paper)
            if self.root_state is not None and s == self.game.get_string_representation(self.root_state):
                num_valid = int(valid_actions.sum())
                if num_valid > 1:  # Only add noise if there are multiple valid actions
                    noise = np.random.dirichlet([self.dirichlet_alpha] * num_valid)
                    noise_full = np.zeros_like(self.Ps[s])
                    noise_full[valid_actions == 1] = noise
                    self.Ps[s] = (1 - self.dirichlet_epsilon) * self.Ps[s] + self.dirichlet_epsilon * noise_full
                    self.Ps[s] /= np.sum(self.Ps[s])  # Renormalize

            self.Cs[s] = valid_actions
            self.Ns[s] = 0  # leaf node has been visited
            return -v

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
        assert a is not None

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

    def run(
        self, start_node: Optional[DraftState] = None, return_action_policy: Optional[bool] = True
    ) -> Optional[np.ndarray]:
        """Run MCTS simulations from start node with optional Dirichlet noise at root.

        Args:
            start_node: Start node for simulations
            return_action_policy: Whether to return probability move vector

        Returns:
            Action policy if return_action_policy is True
        """
        if start_node is None:
            start_node = self.game.get_start_node()

        # Set root state for Dirichlet noise application
        self.root_state = start_node

        if self.search_stop_criterion == "time":
            timeLimit = time.time() + self.search_stop_threshold
            while time.time() < timeLimit:
                self.search(start_node)
        else:
            for _ in range(self.search_stop_threshold):
                self.search(start_node)

        if return_action_policy:
            return self.get_action_policy(start_node)

    def get_action_policy(self, state: DraftState) -> np.ndarray:
        """Returns action probabilities with temperature scaling.

        Args:
            state: Start node

        Returns:
            Vector of action probabilities
        """
        s = self.game.get_string_representation(state)
        counts = np.asarray(
            [
                self.Nsa[(s, a)] if (s, a) in self.Nsa else 0
                for a in range(self.game.get_action_size())
            ]
        )

        if self.temperature == 0:
            # Deterministic: choose best action
            policy = np.zeros_like(counts)
            policy[np.argmax(counts)] = 1.0
            return policy
        elif self.temperature == float('inf'):
            # Uniform over visited actions
            policy = (counts > 0).astype(float)
            return policy / np.sum(policy)
        else:
            # Temperature scaling: counts^(1/temp)
            counts = counts.astype(float)
            counts = np.power(counts, 1.0 / self.temperature)
            return counts / np.sum(counts)

    def get_tree_statistics(self, state: DraftState) -> dict:
        """Get statistics about the search tree from a state.

        Args:
            state: State to analyze

        Returns:
            Dictionary with tree statistics
        """
        s = self.game.get_string_representation(state)

        if s not in self.Ns:
            return {'visits': 0, 'children': 0}

        # Count children (visited actions)
        children = []
        valid_actions = self.Cs[s] if s in self.Cs else self.game.get_valid_actions(state)

        for a in range(self.game.get_action_size()):
            if valid_actions[a] and (s, a) in self.Nsa:
                children.append({
                    'action': a,
                    'visits': self.Nsa[(s, a)],
                    'value': self.Qsa[(s, a)] if (s, a) in self.Qsa else 0.0,
                    'policy_prior': self.Ps[s][a] if s in self.Ps else 0.0,
                })

        # Sort by visits
        children.sort(key=lambda x: x['visits'], reverse=True)

        return {
            'visits': self.Ns[s],
            'num_children': len(children),
            'children': children[:10],  # Top 10 children
            'value_estimates': [c['value'] for c in children[:5]],
            'visit_distribution': [c['visits'] for c in children[:5]],
        }
