from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, LaunchConfiguration

from launch_ros.actions import Node

def generate_launch_description():

    nodes_to_start = []  
    nodes_to_start.append(
        Node(
            package='stack_approach',
            executable='roller_gripper',
            name='right_roller_gripper',
            output="screen",
            parameters=[{
                'finger_port': 1,
                'roller_port': 2,
            }]
        ),
    )
    nodes_to_start.append(
        Node(
            package='stack_approach',
            executable='roller_gripper',
            name='left_roller_gripper',
            output="screen",
            parameters=[{
                'finger_port': 4,
                'roller_port': 3,
            }]
        ),
    )
    return LaunchDescription(nodes_to_start)  