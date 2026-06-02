import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_dim=24, action_dim=2, max_action=1.0):
        super().__init__()

        self.max_action = max_action

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.max_action * self.net(state)