from oracle.drafting.game import BasicDraft
from oracle.drafting.mcts import ClassicMCTS, random_policy

class Coach:
    def __init__(self, draft_game: BasicDraft, a0_nnet) -> None:
        self.draft_game = draft_game
        self.a0_nnet = a0_nnet
        self.mcts = ClassicMCTS(self.draft_game, random_policy)
        
    def self_playthrough(self):
        # start with a new game
        draft_state = self.draft_game.get_start_node()
        
        while True:
            # until the game is finished
            
            self.mcts.search(draft_state)
            
        