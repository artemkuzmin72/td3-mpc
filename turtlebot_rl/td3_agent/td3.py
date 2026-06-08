"""
Key improvements over DDPG
---------------------------
1. Clipped Double-Q  — use min(Q1, Q2) as regression target to cut
                        over-estimation bias.
2. Delayed policy update  — update Actor & target nets every `policy_freq`
                            critic steps so the critic stabilises first.
3. Target policy smoothing  — add clipped Gaussian noise to target actions
                              so the critic cannot over-fit sharp peaks.
"""

import copy
import torch
import torch.nn.functional as F

from .actor         import Actor
from .critic        import Critic
from .replay_buffer import ReplayBuffer


class TD3:
    """
    TD3 agent.

    Parameters
    ----------
    state_dim   : dimensionality of the observation vector (24 for this robot)
    action_dim  : number of action components (2: linear, angular)
    max_action  : absolute upper-bound on each action component (0.4 m/s)
    gamma       : discount factor
    tau         : soft-update coefficient for target networks
    policy_noise: std of smoothing noise added to *target* actions
    noise_clip  : clipping range for smoothing noise
    policy_freq : actor + target update period (in critic update steps)
    lr_actor    : actor learning rate
    lr_critic   : critic learning rate
    """

    def __init__(
        self,
        state_dim:    int,
        action_dim:   int,
        max_action:   float,
        gamma:        float = 0.99,
        tau:          float = 0.005,
        policy_noise: float = 0.2,
        noise_clip:   float = 0.5,
        policy_freq:  int   = 2,
        lr_actor:     float = 3e-4,
        lr_critic:    float = 3e-4,
    ):
        self.max_action   = max_action
        self.gamma        = gamma
        self.tau          = tau
        self.policy_noise = policy_noise
        self.noise_clip   = noise_clip
        self.policy_freq  = policy_freq

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -------- Actor (online + target) --------
        self.actor        = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_opt    = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        # -------- Critic / Twin-Q (online + target) --------
        self.critic        = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_opt    = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # internal step counter for delayed policy update
        self._train_step = 0

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def select_action(self, state) -> "np.ndarray":
        """
        Greedy deterministic action — used for both exploration (with
        external noise added in train_node.py) and evaluation.

        Args:
            state: array-like of shape (state_dim,)
        Returns:
            numpy array of shape (action_dim,)
        """
        import numpy as np
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(s).cpu().numpy().flatten()
        return action

    # ------------------------------------------------------------------
    def train(self, buffer: ReplayBuffer, batch_size: int = 256):
        """
        One training step (one critic update + optional actor update).

        Implements the three TD3 tricks:
            1. Clipped Double-Q target
            2. Target policy smoothing
            3. Delayed actor + target update
        """
        self._train_step += 1
        batch = buffer.sample(batch_size, device=self.device)

        state      = batch["state"]
        action     = batch["action"]
        reward     = batch["reward"]
        next_state = batch["next_state"]
        done       = batch["done"]

        # ---- (1 + 2) Compute TD target with smoothed target policy ----
        with torch.no_grad():
            # target policy smoothing
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # clipped double-Q  →  take minimum to reduce over-estimation
            q1_target, q2_target = self.critic_target(next_state, next_action)
            q_target = reward + (1.0 - done) * self.gamma * torch.min(q1_target, q2_target)

        # ---- Critic update ----
        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_opt.step()

        # ---- (3) Delayed actor + target update ----
        if self._train_step % self.policy_freq == 0:

            # Actor: maximise Q1(s, µ(s))
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_opt.step()

            # Soft-update target networks
            self._soft_update(self.actor_target,  self.actor)
            self._soft_update(self.critic_target, self.critic)

    # ==================================================================
    # SAVE / LOAD
    # ==================================================================

    def save(self, path: str):
        """Save all network weights to *path* (no extension needed)."""
        torch.save({
            "actor":         self.actor.state_dict(),
            "actor_target":  self.actor_target.state_dict(),
            "critic":        self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }, path + ".pt")

    def load(self, path: str):
        """Load weights previously saved with save()."""
        ckpt = torch.load(path + ".pt", map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])

    # ==================================================================
    # PRIVATE HELPERS
    # ==================================================================

    def _soft_update(self, target: torch.nn.Module, online: torch.nn.Module):
        """
        Polyak averaging:  θ_target ← τ·θ_online + (1-τ)·θ_target
        """
        for p_t, p_o in zip(target.parameters(), online.parameters()):
            p_t.data.copy_(self.tau * p_o.data + (1.0 - self.tau) * p_t.data)