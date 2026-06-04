import time
import numpy as np

import rclpy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

from gazebo_msgs.srv import SetEntityState


class GazeboEnv:

    COLLISION_DIST = 0.1
    SPAWN_GRACE_STEPS = 5

    def __init__(self, node):
        self.node = node

        # ===== state =====
        self.scan = None
        self.odom = None

        self.goal_x = 4.0
        self.goal_y = 4.0

        self._steps_since_reset = 0

        # IMPORTANT: must match Gazebo model name exactly
        self.robot_name = "burger"

        # ===== ROS IO =====
        self.node.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.node.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.cmd_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)

        # ===== TELEPORT SERVICE =====
        self.set_state_client = self.node.create_client(
            SetEntityState,
            '/set_entity_state'
        )

        while not self.set_state_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info("Waiting /set_entity_state...")

        self.node.get_logger().info("GazeboEnv READY (TELEPORT MODE)")

        self.teleport_robot()

    # ======================================================
    # CALLBACKS
    # ======================================================

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float32)

        ranges[np.isinf(ranges)] = 3.5
        ranges[np.isnan(ranges)] = 3.5

        self.scan = ranges

        self.node.get_logger().info(f"SCAN = {np.min(self.scan):.3f} m")

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        self.odom = msg

    # ======================================================
    # STATE
    # ======================================================

    def get_state(self):

        if self.scan is None or self.odom is None:
            return np.zeros(24, dtype=np.float32)

        lidar = self.scan.copy()

        # downsample
        if len(lidar) >= 20:
            idx = np.linspace(0, len(lidar) - 1, 20).astype(int)
            lidar = lidar[idx]
        else:
            lidar = np.pad(lidar, (0, 20 - len(lidar)), constant_values=3.5)

        x = self.odom.pose.pose.position.x
        y = self.odom.pose.pose.position.y

        dist = np.sqrt((self.goal_x - x) ** 2 + (self.goal_y - y) ** 2)

        return np.concatenate([
            lidar,
            np.array([x, y, dist, 1.0], dtype=np.float32)
        ])

    # ======================================================
    # STEP
    # ======================================================

    def step(self, action):
        twist = Twist()
        twist.linear.x = float(action[0])
        twist.angular.z = float(action[1])

        self.cmd_pub.publish(twist)
        
        start_state = self.get_state()
        start_dist = start_state[22]

        next_state = self.get_state()
        end_dist = next_state[22]

        reward = 5.0 * (start_dist - end_dist)
        reward -= 10

        done = False

        if self.scan is not None:

            min_dist = np.min(self.scan)

            if min_dist < 0.17:

                self.node.get_logger().warn(
                    "COLLISION -> TELEPORT"
                )

                self.teleport_robot()

                reward -= 50.0
                done = True

        if end_dist < 0.4:

            self.node.get_logger().warn(
                "GOAL REACHED"
            )

            reward += 300.0
            done = True

        return next_state, reward, done
    
    def teleport_robot(self):

        self.cmd_pub.publish(Twist())

        req = SetEntityState.Request()

        req.state.name = "burger"

        req.state.pose.position.x = 0.0
        req.state.pose.position.y = 0.0
        req.state.pose.position.z = 0.05

        req.state.pose.orientation.x = 0.0
        req.state.pose.orientation.y = 0.0
        req.state.pose.orientation.z = 0.0
        req.state.pose.orientation.w = 1.0

        req.state.twist.linear.x = 0.0
        req.state.twist.linear.y = 0.0
        req.state.twist.linear.z = 0.0

        req.state.twist.angular.x = 0.0
        req.state.twist.angular.y = 0.0
        req.state.twist.angular.z = 0.0

        future = self.set_state_client.call_async(req)

        start = time.time()

        while not future.done():

            rclpy.spin_once(self.node, timeout_sec=0.01)
            return

        self.node.get_logger().warn(
            "TELEPORT DONE"
        )

        # Даем времени обновиться odom и scan
        end_time = time.time() + 0.5

        while time.time() < end_time:
            rclpy.spin_once(self.node, timeout_sec=0.01)