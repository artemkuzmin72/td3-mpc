import numpy as np


class ReplayBuffer:
    def __init__(self, max_size=1_000_000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = None
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = None

    def init(self, state_dim, action_dim):
        self.state = np.zeros((self.max_size, state_dim))
        self.next_state = np.zeros((self.max_size, state_dim))
        self.action = np.zeros((self.max_size, action_dim))
        self.reward = np.zeros((self.max_size, 1))
        self.done = np.zeros((self.max_size, 1))

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)

        return (
            self.state[idx],
            self.action[idx],
            self.reward[idx],
            self.next_state[idx],
            self.done[idx]
        )