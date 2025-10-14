"""Performance benchmarking utilities."""

import time
from typing import Callable

import numpy as np

from oracle.drafting.game import DraftState, Game
from oracle.drafting.mcts import BaseMCTS


def benchmark_mcts_speed(
    mcts: BaseMCTS,
    game: Game,
    num_positions: int = 10,
    simulations_per_position: int = 100
) -> dict:
    """Benchmark MCTS search speed.

    Args:
        mcts: MCTS instance
        game: Game instance
        num_positions: Number of different positions to test
        simulations_per_position: Simulations per position

    Returns:
        Dictionary with benchmark results
    """
    total_simulations = 0
    total_time = 0

    for _ in range(num_positions):
        # Generate random position
        state = game.get_start_node()
        num_moves = np.random.randint(0, 5)
        for _ in range(num_moves):
            valid_actions = game.get_valid_actions(state)
            action = np.random.choice(np.where(valid_actions)[0])
            state = game.take_action(state, action)

        # Reset MCTS
        mcts.reset()

        # Time the search
        start_time = time.time()
        mcts.search_stop_threshold = simulations_per_position
        mcts.search_stop_criterion = "simulations"

        for _ in range(simulations_per_position):
            mcts.search(state)

        elapsed = time.time() - start_time

        total_simulations += simulations_per_position
        total_time += elapsed

    results = {
        'total_simulations': total_simulations,
        'total_time': total_time,
        'simulations_per_second': total_simulations / total_time,
        'seconds_per_simulation': total_time / total_simulations,
        'average_time_per_position': total_time / num_positions,
    }

    return results


def benchmark_model_inference(
    predict_fn: Callable,
    num_heroes: int = 112,
    batch_sizes: list = None,
    num_iterations: int = 100
) -> dict:
    """Benchmark neural network inference speed.

    Args:
        predict_fn: Function that takes draft array and returns (policy, value)
        num_heroes: Number of heroes
        batch_sizes: List of batch sizes to test
        num_iterations: Number of iterations per batch size

    Returns:
        Dictionary with benchmark results per batch size
    """
    if batch_sizes is None:
        batch_sizes = [1, 8, 32, 128, 512]

    results = {}

    for batch_size in batch_sizes:
        # Generate random drafts
        drafts = np.random.choice([-1, 0, 1], size=(batch_size, num_heroes)).astype(np.float32)

        # Warmup
        for _ in range(5):
            predict_fn(drafts[0] if batch_size == 1 else drafts)

        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            predict_fn(drafts[0] if batch_size == 1 else drafts)
        elapsed = time.time() - start_time

        samples_processed = batch_size * num_iterations

        results[f'batch_{batch_size}'] = {
            'time_per_batch': elapsed / num_iterations,
            'samples_per_second': samples_processed / elapsed,
            'ms_per_sample': (elapsed / samples_processed) * 1000,
        }

    return results


def compare_policy_quality(
    mcts_policy: np.ndarray,
    network_policy: np.ndarray,
    valid_actions: np.ndarray
) -> dict:
    """Compare MCTS policy with raw network policy.

    Args:
        mcts_policy: Policy from MCTS search (num_actions,)
        network_policy: Policy from network (num_actions,)
        valid_actions: Binary mask of valid actions (num_actions,)

    Returns:
        Comparison metrics
    """
    # Mask to valid actions only
    mcts_valid = mcts_policy[valid_actions == 1]
    network_valid = network_policy[valid_actions == 1]

    # Normalize
    mcts_valid = mcts_valid / mcts_valid.sum()
    network_valid = network_valid / network_valid.sum()

    # KL divergence: MCTS || Network
    kl_div = np.sum(mcts_valid * np.log((mcts_valid + 1e-8) / (network_valid + 1e-8)))

    # Top-1 agreement
    top1_mcts = np.argmax(mcts_valid)
    top1_network = np.argmax(network_valid)
    top1_agree = (top1_mcts == top1_network)

    # Top-k overlap
    k = min(5, len(mcts_valid))
    topk_mcts = set(np.argsort(mcts_valid)[-k:])
    topk_network = set(np.argsort(network_valid)[-k:])
    topk_overlap = len(topk_mcts & topk_network) / k

    return {
        'kl_divergence': kl_div,
        'top1_agreement': top1_agree,
        f'top{k}_overlap': topk_overlap,
    }
