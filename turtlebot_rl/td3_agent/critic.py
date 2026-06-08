import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()

        in_dim = state_dim + action_dim

        # -------- Q1 --------
        self.q1 = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        # -------- Q2 --------
        self.q2 = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns both Q-values.

        Args:
            state:  (batch, state_dim)
            action: (batch, action_dim)
        Returns:
            q1: (batch, 1)
            q2: (batch, 1)
        """
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)

    # ------------------------------------------------------------------
    def Q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Returns Q1 only — used for the actor (policy) gradient update.
        """
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa)