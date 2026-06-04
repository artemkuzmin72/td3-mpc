import os
import numpy as np
import torch

import rclpy
from rclpy.node import Node

from turtlebot_rl.env.env_node import GazeboEnv
from turtlebot_rl.td3_agent.td3 import TD3
from turtlebot_rl.td3_agent.replay_buffer import ReplayBuffer


class TrainNode(Node):

    def __init__(self):

        super().__init__('td3_train_node')

        # ===== ENV =====

        self.env = GazeboEnv(self)

        # ===== TD3 =====

        self.state_dim = 24

        self.action_dim = 2

        self.max_action = 0.4

        self.agent = TD3(

            state_dim=self.state_dim,

            action_dim=self.action_dim,

            max_action=self.max_action

        )

        # ===== REPLAY BUFFER =====

        self.buffer = ReplayBuffer()

        self.buffer.init(self.state_dim, self.action_dim)

        # ===== TRAIN PARAMS =====

        self.batch_size = 256

        self.warmup_steps = 2000

        self.train_freq = 1

        self.best_reward = -1e9

        # ===== STATE =====

        self.state = self.env.get_state()

        self.episode_reward = 0

        self.episode = 0

        self.total_steps = 0

        self.done = False

        # ===== LOOP =====

        self.timer = self.create_timer(0.1, self.loop)

        self.get_logger().info("TD3 TrainNode started")

    # --------------------------------------------------

    # POLICY STEP

    # --------------------------------------------------

    def act(self, state):
        action = self.agent.select_action(state)

        noise = np.random.normal(0, 0.2, size=2)
        action = action + noise

        action = np.tanh(action)  # стабилизация

        # map properly
        linear = (action[0] + 1) / 2 * 0.4
        angular = action[1]

        return np.array([linear, angular])

    # --------------------------------------------------

    # MAIN LOOP

    # --------------------------------------------------

    def loop(self):
        if self.done:

            self.get_logger().warn(
                f"EPISODE {self.episode} | reward={self.episode_reward:.2f}"
            )

            if self.episode_reward > self.best_reward:

                self.best_reward = self.episode_reward

                self.get_logger().info(
                    f"NEW BEST REWARD = {self.best_reward:.2f}"
                )

            self.episode_reward = 0.0

            self.episode += 1

            self.done = False

            self.state = self.env.get_state()

            return

        # ===== ACTION =====

        action = self.act(self.state)

        # ===== STEP ENV =====

        next_state, reward, done = self.env.step(action)

        # ===== STORE TRANSITION =====

        self.buffer.add(

            self.state,

            action,

            reward,

            next_state,

            float(done)

        )

        self.state = next_state

        self.done = done

        self.episode_reward += reward

        self.total_steps += 1

        # ===== TRAIN TD3 =====

        if self.buffer.size > self.warmup_steps and self.total_steps % self.train_freq == 0:

            self.agent.train(self.buffer, self.batch_size)

        # ===== OPTIONAL LOG =====

        if self.total_steps % 200 == 0:

            self.get_logger().info(

                f"steps={self.total_steps} buffer={self.buffer.size}"

            )

# --------------------------------------------------

# MAIN

# --------------------------------------------------

def main():

    rclpy.init()

    node = TrainNode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()

if __name__ == '__main__':

    main()