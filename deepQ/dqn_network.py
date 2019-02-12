import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_num=64, fc2_num=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        # Use linear fd for learning purpose
        self.fc1 = nn.Linear(state_size, fc1_num)
        self.fc2 = nn.Linear(fc1_num, fc2_num)
        self.fc3 = nn.Linear(fc2_num, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        print(state)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
