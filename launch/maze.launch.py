import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    world = os.path.join(
        get_package_share_directory('turtlebot_rl'),
        'worlds',
        'maze.world'
    )

    # 1. Gazebo (EMPTY + your world)
    gazebo = ExecuteProcess(
        cmd=['gazebo', '--verbose', world,
             '-s', 'libgazebo_ros_init.so',
             '-s', 'libgazebo_ros_factory.so'],
        output='screen'
    )

    # 3. spawn robot from topic (IMPORTANT)
    spawn = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'burger',
            '-file', '/opt/ros/humble/share/turtlebot3_gazebo/models/turtlebot3_burger/model.sdf',
            '-x', '0',
            '-y', '0',
            '-z', '0.1'
        ],
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        spawn
    ])