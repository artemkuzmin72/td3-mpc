import rclpy
import numpy as np
import torch
import os
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from turtlebot_rl.td3_agent.td3 import TD3

from turtlebot_rl.mpc_controller import (
    MPCController
)

from turtlebot_rl.gazebo_env import (
    GazeboEnv
)


pkg_path = get_package_share_directory('turtlebot_rl')
model_path = os.path.join(pkg_path, 'weights', 'actor.pth')


class RLNode(Node):

    def __init__(self):

        super().__init__('td3_rl_node')

        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.agent = TD3(
            state_dim=24,
            action_dim=2,
            max_action=1.0
        )

        self.agent.actor.load_state_dict(
            torch.load(
                model_path,
                map_location='cpu'
            )
        )

        self.agent.actor.eval()

        self.mpc = MPCController()

        self.env = GazeboEnv()

        self.lidar = np.zeros(20)

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.timer = self.create_timer(
            0.1,
            self.control_loop
        )

    def scan_callback(self, msg):

        self.lidar = np.array(
            msg.ranges[:20]
        )

    def odom_callback(self, msg):

        self.x = (
            msg.pose.pose.position.x
        )

        self.y = (
            msg.pose.pose.position.y
        )

    def control_loop(self):

        state = self.env.build_state(
            self.lidar,
            self.x,
            self.y,
            self.yaw
        )

        action = self.agent.select_action(
            state
        )

        safe_action = (
            self.mpc.correct_action(action)
        )

        twist = Twist()

        twist.linear.x = float(
            safe_action[0]
        )

        twist.angular.z = float(
            safe_action[1]
        )

        self.cmd_pub.publish(twist)


def main(args=None):

    rclpy.init(args=args)

    node = RLNode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':

    main()
