"""Demo: Use trained AlphaZero model for draft recommendations.

This script demonstrates how to load a trained AlphaZero model and use it
to recommend hero picks during a draft.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from oracle.config import AlphaZeroConfig
from oracle.drafting.alpha_zero.neural_net import NeuralNetWrapper, create_alphazero_net
from oracle.drafting.game import BasicDraft, DraftState
from oracle.drafting.mcts import AlphaZeroMCTS
from oracle.utils.logger import getLogger

logger = getLogger(__name__)

# Hero ID to name mapping (simplified for demo)
HERO_NAMES = {
    0: "Antimage",
    1: "Axe",
    2: "Bane",
    3: "Bloodseeker",
    4: "Crystal Maiden",
    5: "Drow Ranger",
    6: "Earthshaker",
    7: "Juggernaut",
    8: "Mirana",
    9: "Morphling",
    10: "Shadow Fiend",
}


class DraftRecommender:
    """Interactive draft recommendation system."""

    def __init__(self, model_path: Path, config: AlphaZeroConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = create_alphazero_net(
            num_heroes=config.num_heroes,
            embedding_dim=config.embedding_dim,
            hero_hidden_dim=config.hero_hidden_dim,
            num_attention_heads=config.num_attention_heads,
            draft_hidden_dim=config.draft_hidden_dim,
            dropout=0.0,  # No dropout for inference
        )

        self.net_wrapper = NeuralNetWrapper(self.model, device=self.device)
        self.net_wrapper.load_checkpoint(model_path)
        logger.info("Model loaded successfully")

        # Setup game and MCTS
        self.game = BasicDraft(reward_model=self._net_reward)
        self.mcts = AlphaZeroMCTS(
            drafting_game=self.game,
            game_net=self.net_wrapper.predict,
            search_stop_criterion="simulations",
            search_number_of_simulations=config.num_simulations,
            cpuct=config.cpuct,
        )

    def _net_reward(self, draft: np.ndarray) -> float:
        """Use network value head for position evaluation."""
        _, value = self.net_wrapper.predict(draft)
        return value

    def get_recommendations(
        self, draft_state: DraftState, top_k: int = 5
    ) -> list[tuple[int, float]]:
        """Get top-k hero recommendations for current draft state.

        Args:
            draft_state: Current draft state
            top_k: Number of recommendations to return

        Returns:
            List of (hero_id, probability) tuples
        """
        # Run MCTS to get policy
        policy = self.mcts.run(draft_state, return_action_policy=True)

        # Get valid actions
        valid_actions = self.game.get_valid_actions(draft_state)

        # Filter to valid actions and get top-k
        recommendations = []
        for hero_id in range(len(policy)):
            if valid_actions[hero_id] == 1:
                recommendations.append((hero_id, policy[hero_id]))

        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_k]

    def get_position_evaluation(self, draft_state: DraftState) -> float:
        """Get network's evaluation of current position.

        Returns:
            Value in [-1, 1] representing advantage
            Positive = Radiant advantage, Negative = Dire advantage
        """
        _, value = self.net_wrapper.predict(draft_state)
        return value


def demo_interactive(recommender: DraftRecommender):
    """Interactive demo where user builds a draft step-by-step."""
    logger.info("\n" + "=" * 60)
    logger.info("Interactive Draft Recommendation Demo")
    logger.info("=" * 60)

    draft_state = recommender.game.get_start_node()
    move_num = 0

    while not recommender.game.is_game_terminated(draft_state):
        move_num += 1
        player = "Radiant" if draft_state.player == 0 else "Dire"

        logger.info(f"\nMove {move_num} - {player}'s turn")

        # Show current draft
        radiant_picks = np.where(draft_state.draft == 1)[0].tolist()
        dire_picks = np.where(draft_state.draft == -1)[0].tolist()

        logger.info(f"Radiant picks: {radiant_picks}")
        logger.info(f"Dire picks: {dire_picks}")

        # Get recommendations
        logger.info("\nGetting recommendations (running MCTS)")
        recommendations = recommender.get_recommendations(draft_state, top_k=5)

        logger.info(f"\nTop 5 recommended heroes for {player}:")
        for i, (hero_id, prob) in enumerate(recommendations, 1):
            hero_name = HERO_NAMES.get(hero_id, f"Hero_{hero_id}")
            logger.info(f"  {i}. {hero_name:20s} (ID: {hero_id:3d})  - {prob*100:5.2f}%")

        # Get position evaluation
        value = recommender.get_position_evaluation(draft_state)
        logger.info(f"\nPosition evaluation: {value:+.3f}")
        if abs(value) < 0.1:
            logger.info("  → Balanced position")
        elif value > 0:
            logger.info("  → Radiant advantage")
        else:
            logger.info("  → Dire advantage")

        # Auto-pick top recommendation for demo
        chosen_hero = recommendations[0][0]
        hero_name = HERO_NAMES.get(chosen_hero, f"Hero_{chosen_hero}")
        logger.info(f"\n{player} picks: {hero_name} (ID: {chosen_hero})")

        # Update state
        draft_state = recommender.game.take_action(draft_state, chosen_hero)

        if move_num >= 10:  # Stop after 10 moves for demo
            break

    logger.info("\n" + "=" * 60)
    logger.info("Demo complete!")
    logger.info("=" * 60)


def demo_analyze_draft(recommender: DraftRecommender):
    """Analyze a specific draft composition."""
    logger.info("\n" + "=" * 60)
    logger.info("Draft Analysis Demo")
    logger.info("=" * 60)

    # Example draft
    draft = np.zeros(112)
    radiant_picks = [0, 6, 7, 10, 15]  # Example heroes
    dire_picks = [1, 2, 8, 12, 20]

    for hero_id in radiant_picks:
        draft[hero_id] = 1
    for hero_id in dire_picks:
        draft[hero_id] = -1

    logger.info(f"\nAnalyzing draft:")
    logger.info(f"Radiant: {radiant_picks}")
    logger.info(f"Dire: {dire_picks}")

    # Get evaluation
    draft_state = DraftState(draft=draft, player=0)
    value = recommender.get_position_evaluation(draft_state)

    logger.info(f"\nNetwork evaluation: {value:+.3f}")
    radiant_win_prob = (value + 1) / 2 * 100
    logger.info(f"Estimated Radiant win probability: {radiant_win_prob:.1f}%")
    logger.info(f"Estimated Dire win probability: {100-radiant_win_prob:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Demo AlphaZero draft recommendations")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("checkpoints/alphazero/best_model.pt"),
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=200,
        help="MCTS simulations for recommendations",
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "analyze"],
        default="interactive",
        help="Demo mode",
    )

    args = parser.parse_args()

    # Check model exists
    if not args.model_path.exists():
        logger.error(f"Model not found: {args.model_path}")
        logger.error("Train a model first with: python scripts/train_alphazero.py")
        return

    # Create config
    config = AlphaZeroConfig(num_simulations=args.simulations)

    # Create recommender
    recommender = DraftRecommender(args.model_path, config)

    # Run demo
    if args.mode == "interactive":
        demo_interactive(recommender)
    else:
        demo_analyze_draft(recommender)


if __name__ == "__main__":
    main()
