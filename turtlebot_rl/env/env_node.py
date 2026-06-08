import math
import time
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import SetEntityState


class GazeboEnv:
    COLLISION_DIST = 0.17
    GRACE_STEPS    = 8
    LIDAR_N        = 20
    LIDAR_MAX      = 3.5
    GOAL_THRESHOLD = 0.5
    PHYSICS_WAIT   = 0.15
    TELEPORT_SLEEP = 0.5

    def __init__(self, node: Node, goal_x: float = 4.0, goal_y: float = 4.0):
        self.node   = node
        self.goal_x = goal_x
        self.goal_y = goal_y

        self.scan: np.ndarray | None = None
        self.odom = None
        self._steps_since_reset = 0
        self._prev_dist = math.sqrt(goal_x**2 + goal_y**2)

        self.node.create_subscription(LaserScan, '/scan', self._scan_cb, 10)
        self.node.create_subscription(Odometry,  '/odom', self._odom_cb, 10)
        self.cmd_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)

        # Сервис — просто создаём клиент, не ждём ответа при вызове
        self._teleport_client = self.node.create_client(
            SetEntityState, '/set_entity_state')

        while not self._teleport_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info("Waiting for /set_entity_state ...")

        self.node.get_logger().info("GazeboEnv READY")
        self._teleport(0.0, 0.0)

    # ==================================================================
    # CALLBACKS
    # ==================================================================
    def _scan_cb(self, msg: LaserScan):
        r = np.array(msg.ranges, dtype=np.float32)
        r[~np.isfinite(r)] = self.LIDAR_MAX
        self.scan = r

    def _odom_cb(self, msg: Odometry):
        self.odom = msg

    # ==================================================================
    # PUBLIC API
    # ==================================================================
    def reset(self) -> np.ndarray:
        self._teleport(0.0, 0.0)
        self._steps_since_reset = 0
        return self.get_state()

    def get_state(self) -> np.ndarray:
        if self.scan is None or self.odom is None:
            return np.zeros(24, dtype=np.float32)

        lidar = self.scan.copy()
        if len(lidar) >= self.LIDAR_N:
            idx   = np.linspace(0, len(lidar) - 1, self.LIDAR_N).astype(int)
            lidar = lidar[idx]
        else:
            lidar = np.pad(lidar, (0, self.LIDAR_N - len(lidar)),
                           constant_values=self.LIDAR_MAX)
        lidar_norm = np.clip(lidar / self.LIDAR_MAX, 0.0, 1.0)

        x   = self.odom.pose.pose.position.x
        y   = self.odom.pose.pose.position.y
        qz  = self.odom.pose.pose.orientation.z
        qw  = self.odom.pose.pose.orientation.w
        yaw = 2.0 * math.atan2(qz, qw)

        dx   = self.goal_x - x
        dy   = self.goal_y - y
        dist = math.sqrt(dx * dx + dy * dy)
        angle_to_goal = math.atan2(math.sin(math.atan2(dy, dx) - yaw),
                                   math.cos(math.atan2(dy, dx) - yaw))

        return np.concatenate([
            lidar_norm,
            np.array([
                dist / 10.0,
                math.sin(angle_to_goal),
                math.cos(angle_to_goal),
                float(np.min(lidar_norm)),
            ], dtype=np.float32)
        ]).astype(np.float32)

    def step(self, action: np.ndarray):
        self._steps_since_reset += 1

        twist = Twist()
        twist.linear.x  = float(action[0])
        twist.angular.z = float(action[1])
        self.cmd_pub.publish(twist)

        time.sleep(self.PHYSICS_WAIT)

        next_state = self.get_state()
        dist      = next_state[20] * 10.0
        min_lidar = next_state[23] * self.LIDAR_MAX

        prev_dist       = self._prev_dist
        self._prev_dist = dist

        reward = 5.0 * (prev_dist - dist) - 0.3
        if min_lidar < 0.5:
            reward -= 0.5 * (0.5 - min_lidar) / 0.5

        done = False

        if dist < self.GOAL_THRESHOLD:
            self.node.get_logger().warn(f"GOAL REACHED  dist={dist:.3f}m")
            reward += 300.0
            done    = True
            self._teleport(0.0, 0.0)
            self._steps_since_reset = 0

        elif (self._steps_since_reset > self.GRACE_STEPS
                and min_lidar < self.COLLISION_DIST):
            self.node.get_logger().warn(f"COLLISION  min={min_lidar:.3f}m")
            self.cmd_pub.publish(Twist())
            reward -= 50.0
            done    = True
            self._teleport(0.0, 0.0)
            self._steps_since_reset = 0

        return next_state, reward, done

    # ==================================================================
    # ТЕЛЕПОРТ — fire-and-forget, не ждём ответа сервиса
    # ==================================================================
    def _teleport(self, x: float = 0.0, y: float = 0.0):
        self.cmd_pub.publish(Twist())  # стоп

        req = SetEntityState.Request()
        req.state.name = "burger"
        req.state.pose.position.x = x
        req.state.pose.position.y = y
        req.state.pose.position.z = 0.05
        req.state.pose.orientation.w = 1.0

        self._teleport_client.call_async(req)

        # Ждём фиксированное время пока Gazebo переместит робота
        # и придут свежие данные сенсоров
        time.sleep(self.TELEPORT_SLEEP)

        self._prev_dist = math.sqrt(
            (self.goal_x - x)**2 + (self.goal_y - y)**2
        )
        self.node.get_logger().info(f"TELEPORT -> ({x:.1f}, {y:.1f})")
