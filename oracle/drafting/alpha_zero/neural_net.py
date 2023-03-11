import torch
import torch.nn as nn
from torch import optim

from oracle.drafting.game import BasicDraft

class DraftingConvNet(nn.Module):
    def __init__(self, game: BasicDraft, args) -> None:
        super().__init__() 
        
        self.game = game
        self.args = args
        self.action_size = self.game.get_action_size()
        
        # Output height = (Input height + padding height top + padding height bottom - kernel height) / (stride height) + 1
        self.conv1 = nn.Conv1d(1, 55, 4, stride=2)  # input = 112, output = (112 - 4) / 2 + 1 = 55
        self.conv2 = nn.Conv1d(55, 26, kernel_size=5, stride=2)  # input = 55, output = (55 - 5) / 2 + 1 = 26
        
        self.bn1 = nn.BatchNorm1d(55)
        self.bn2 = nn.BatchNorm1d(26)
        
        self.fc1 = nn.Linear(26, self.action_size)
        self.fc2 = nn.Linear(self.action_size, 1)
        
        self.policy_net = nn.Sequential(
            self.conv1,
            self.bn1,
            self.conv2,
            self.bn2,
            self.fc1,
        )
        
        self.policy = nn.Sequential(
            self.policy_net,
            nn.Softmax()
        )
        
        self.value = nn.Sequential(
            self.policy_net,
            self.fc2,
            nn.Tanh()
        )
        
    def forward(self, batch):
        
        pi = self.policy(batch)
        value = self.value(batch)
        return pi, value
        
        

if __name__ == "__main__":
    import numpy as np
    test_draft = np.zeros((112, 1))
    choices = np.random.choice(112, 10, replace=False)
    test_draft[choices[:5]] = +1
    test_draft[choices[:5]] = -1
    
    from oracle.drafting.game import BasicDraft
    
    game = BasicDraft(lambda x: np.random.rand())
    
    net = DraftingConvNet(game, dict())
    
    print("1")
    