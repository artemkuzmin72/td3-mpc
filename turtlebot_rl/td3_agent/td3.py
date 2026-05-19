import copy
import torch
import torch.nn.functional as F
import torch.optim as optim

from turtlebot_rl.td3_agent.actor import Actor
from turtlebot_rl.td3_agent.critic import Critic


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


class TD3:

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action
    ):

        self.actor = Actor(
            state_dim,
            action_dim,
            max_action
        ).to(device)

        self.actor_target = copy.deepcopy(
            self.actor
        )

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=3e-4
        )

        self.critic_1 = Critic(
            state_dim,
            action_dim
        ).to(device)

        self.critic_2 = Critic(
            state_dim,
            action_dim
        ).to(device)

        self.critic_target_1 = copy.deepcopy(
            self.critic_1
        )

        self.critic_target_2 = copy.deepcopy(
            self.critic_2
        )

        self.critic_optimizer_1 = optim.Adam(
            self.critic_1.parameters(),
            lr=3e-4
        )

        self.critic_optimizer_2 = optim.Adam(
            self.critic_2.parameters(),
            lr=3e-4
        )

        self.max_action = max_action

    def select_action(self, state):

        state = torch.FloatTensor(
            state.reshape(1, -1)
        ).to(device)

        return (
            self.actor(state)
            .cpu()
            .data
            .numpy()
            .flatten()
        )
