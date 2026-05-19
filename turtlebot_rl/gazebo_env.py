import numpy as np


class GazeboEnv:

    def __init__(self):

        self.state_dim = 24

    def build_state(
        self,
        lidar,
        x,
        y,
        yaw
    ):

        state = np.zeros(24)

        lidar = np.array(lidar[:20])

        state[:20] = lidar

        state[20] = x
        state[21] = y
        state[22] = yaw

        distance_to_goal = np.sqrt(
            (5 - x) ** 2 +
            (5 - y) ** 2
        )

        state[23] = distance_to_goal

        return state
