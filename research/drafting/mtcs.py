import time
import math
import random


def random_policy(state) -> float:
    """
    Random policy for Monte Carlo Tree Search.
    Randomly selects an action from the possible choices
    for game playout.

    This is the most basic form of playout.
    """
    while not state.is_terminal():
        try:
            action = random.choice(state.get_possible_actions())
        except IndexError as exc:
            raise Exception(
                f"Non-terminal state has no possible actions: {state}"
            ) from exc
        state = state.take_action(action)
    return state.get_reward()


class TreeNode:
    def __init__(self, state, parent):
        self.state = state
        self.is_terminal = state.is_terminal()
        self.is_fully_expanded = self.is_terminal
        self.parent = parent
        self.num_visits = 0
        self.total_reward = 0
        self.children = {}

    def __str__(self):
        s = []
        s.append(f"totalReward: {self.total_reward}")
        s.append(f"numVisits: {self.num_visits}")
        s.append(f"isTerminal: {self.is_terminal}")
        s.append(f"possibleActions: {self.children.keys()}")
        return "%s: {%s}" % (self.__class__.__name__, ", ".join(s))


class MonteCarloTreeSearch:
    def __init__(
        self,
        time_limit=None,
        iteration_limit=None,
        exploration_constant=1 / math.sqrt(2),
        rollout_policy=random_policy,
    ):
        if time_limit is not None:
            if iteration_limit is not None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.time_limit = time_limit
            self.limit_type = "time"
        else:
            if iteration_limit is None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iteration_limit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.search_limit = iteration_limit
            self.limit_type = "iterations"
        self.exploration_constant = exploration_constant
        self.rollout = rollout_policy

    def search(self, initial_state, need_details=False):
        self.root = TreeNode(initial_state, None)

        if self.limit_type == "time":
            timeLimit = time.time() + self.time_limit / 1000
            while time.time() < timeLimit:
                self.execute_round()
        else:
            for i in range(self.search_limit):
                self.execute_round()

        best_child = self.get_best_child(self.root, 0)
        action = (
            action for action, node in self.root.children.items() if node is best_child
        ).__next__()
        if need_details:
            return {
                "action": action,
                "expectedReward": best_child.total_reward / best_child.num_visits,
            }
        return action

    def execute_round(self):
        """
        execute a selection-expansion-simulation-backpropagation round
        """
        node = self.select_node(self.root)
        reward = self.rollout(node.state)
        self.backpropogate(node, reward)

    def select_node(self, node):
        while not node.is_terminal:
            if node.is_fully_expanded:
                node = self.get_best_child(node, self.exploration_constant)
            else:
                return self.expand(node)
        return node

    def expand(self, node):
        actions = node.state.get_possible_actions()
        for action in actions:
            if action not in node.children:
                new_node = TreeNode(node.state.take_action(action), node)
                node.children[action] = new_node
                if len(actions) == len(node.children):
                    node.is_fully_expanded = True
                return new_node

        raise Exception("Should never reach here")

    def backpropogate(self, node, reward):
        while node is not None:
            node.num_visits += 1
            node.total_reward += reward
            node = node.parent

    def get_best_child(self, node, exploration_value):
        best_value = float("-inf")
        best_nodes = []
        for child in node.children.values():
            node_value = (
                node.state.get_current_player() * child.total_reward / child.num_visits
                + exploration_value
                * math.sqrt(2 * math.log(node.num_visits) / child.num_visits)
            )
            if node_value > best_value:
                best_value = node_value
                best_nodes = [child]
            elif node_value == best_value:
                best_nodes.append(child)
        return random.choice(best_nodes)
