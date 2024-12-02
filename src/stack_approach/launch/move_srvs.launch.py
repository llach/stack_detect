from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, LaunchConfiguration

from launch_ros.actions import Node

def generate_launch_description():
    # Declare arguments
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            "sim",
            default_value="false",
            description="are we in sim?",
        )
    )

    # Initialize Arguments
    sim = LaunchConfiguration("sim")

    nodes_to_start = []
    nodes_to_start.append(
        Node(
            package='stack_approach',
            executable='move_arm_service',
            name='move_arm_service',
            output="screen"
        ),
    )
    nodes_to_start.append(
        Node(
            package='stack_approach',
            executable='gripper',
            name='gripper',
            output="screen",
            parameters=[{
                'sim': sim
            }]
        ),
    )
    return LaunchDescription(declared_arguments + nodes_to_start)  