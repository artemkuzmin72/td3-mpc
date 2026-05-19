import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):

    def __init__(
        self,
        state_dim,
        action_dim
    ):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(
            state_dim + action_dim,
            256
        )

        self.fc2 = nn.Linear(256, 256)

        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):

        x = torch.cat(
            [state, action],
            dim=1
        )

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        q = self.fc3(x)

        return q