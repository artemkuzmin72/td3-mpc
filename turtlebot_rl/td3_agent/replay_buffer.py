import random
import numpy as np


class ReplayBuffer:

    def __init__(self, max_size=100000):

        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, transition):

        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (
                self.ptr + 1
            ) % self.max_size

        else:
            self.storage.append(transition)

    def sample(self, batch_size):

        ind = np.random.randint(
            0,
            len(self.storage),
            size=batch_size
        )

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for i in ind:

            s, a, r, ns, d = self.storage[i]

            states.append(np.array(s))
            actions.append(np.array(a))
            rewards.append(np.array(r))
            next_states.append(np.array(ns))
            dones.append(np.array(d))

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
