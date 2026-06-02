import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

from turtlebot_rl.td3_agent.actor import Actor
from turtlebot_rl.td3_agent.critic import Critic


class TD3:
    def __init__(self, state_dim=24, action_dim=2, max_action=1.0):

        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action

        self.gamma = 0.99
        self.tau = 0.005

        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_delay = 2

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.actor(state).detach().numpy()[0]

    def train(self, replay_buffer, batch_size=256):

        self.total_it += 1

        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done)

        # noise for target policy
        noise = torch.randn_like(action) * self.policy_noise
        noise = noise.clamp(-self.noise_clip, self.noise_clip)

        next_action = self.actor_target(next_state)
        next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

        target_q1, target_q2 = self.critic_target(next_state, next_action)
        target_q = torch.min(target_q1, target_q2)
        target_q = reward + (1 - done) * self.gamma * target_q.detach()

        current_q1, current_q2 = self.critic(state, action)

        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # delayed policy update
        if self.total_it % self.policy_delay == 0:

            actor_loss = -self.critic.q1(
                state,
                self.actor(state)
            ).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # soft update
            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critic, self.critic_target)

    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )