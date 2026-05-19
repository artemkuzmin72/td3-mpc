import numpy as np


class MPCController:

    def __init__(self):

        self.max_linear = 0.22
        self.max_angular = 2.84

    def correct_action(self, action):

        linear = np.clip(
            action[0],
            -self.max_linear,
            self.max_linear
        )

        angular = np.clip(
            action[1],
            -self.max_angular,
            self.max_angular
        )

        return np.array([
            linear,
            angular
        ])
