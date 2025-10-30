from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, LaunchConfiguration

from launch_ros.actions import Node

def generate_launch_description():

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
    nodes_to_start.append(
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            arguments="--yaw 3.1415 --frame-id map --child-frame-id world".split(" "),
            output="both",
        ),
    )
    nodes_to_start.append(
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            arguments="--x -0.021 --z 0.18 --yaw -0.2094 --frame-id right_arm_wrist_3_link --child-frame-id right_finger".split(" "),
            output="both",
        )
    )
    nodes_to_start.append(
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            arguments="--frame-id right_arm_l515_link --child-frame-id camera_link".split(" "),
            output="both",
        )
    )

    return LaunchDescription(nodes_to_start)  