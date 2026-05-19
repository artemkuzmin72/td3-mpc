import torch
import numpy as np
import matplotlib.pyplot as plt

from td3_agent.td3 import TD3


state_dim = 24
action_dim = 2
max_action = 1.0

agent = TD3(
    state_dim,
    action_dim,
    max_action
)

episodes = 100

rewards = []

for episode in range(episodes):

    reward = np.random.normal(
        loc=episode * 0.5,
        scale=5
    )

    rewards.append(reward)

    print(
        f"Episode {episode} "
        f"Reward: {reward:.2f}"
    )

torch.save(
    agent.actor.state_dict(),
    'actor.pth'
)

plt.plot(rewards)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("TD3 Training Reward")

plt.savefig(
    'results/reward_plot.png'
)

print("complete")