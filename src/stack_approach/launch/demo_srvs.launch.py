from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

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
            package='rexp',
            executable='test_srv',
            name='test_srv',
            output="screen",
            condition=IfCondition(sim),
        ),
    )
    nodes_to_start.append(
        Node(
            package='stack_detect',
            executable='cloud_normals',
            name='cloud_normals',
            output="screen",
            condition=UnlessCondition(sim),
        ),
    )
    nodes_to_start.append(
        Node(
            package='stack_approach',
            executable='cloud_pose_vary',
            name='cloud_pose_vary',
            output="screen",
        ),
    )
    nodes_to_start.append(
        Node(
            package='stack_approach',
            executable='collect_cloud_action',
            name='collect_cloud_action',
            output="screen",
            parameters=[{
                'sim': sim
            }]
        ),
    )

    nodes_to_start.append(
        Node(
            package='stack_detect',
            executable='sam2_dino_srv',
            name='sam2_dino_srv',
            output="screen",
        ),
    )


    nodes_to_start.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([PathJoinSubstitution(
                [FindPackageShare("stack_approach"), "launch", "move_srvs.launch.py"]
            )]),
            launch_arguments={
                "sim": sim,
            }.items(),
        )
    )

    return LaunchDescription(declared_arguments + nodes_to_start)  