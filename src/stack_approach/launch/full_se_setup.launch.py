from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, LaunchConfiguration

from launch_ros.actions import Node

def generate_launch_description():

    nodes_to_start = []
    nodes_to_start.append(
        Node(
            package='iri_ur5e_description',
            executable='gown_grasping',
            name='move_dual_service',
            output="screen"
        ),
    )
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

    nodes_to_start.append(
        DeclareLaunchArgument('port', default_value='8383', description='Web server port')
    )

    nodes_to_start.append(
        SetEnvironmentVariable(name='PORT', value=LaunchConfiguration('port'))
    )


    nodes_to_start.append(
        Node(
            package='softenable_display',
            executable='tts_service',
            name='tts_service',
            output='screen'
        )
    )
    nodes_to_start.append(
        Node(
            package='softenable_display',
            executable='change_display',
            name='change_display',
            output='screen'
        )
    )
    nodes_to_start.append(
        Node(
            package='softenable_display',
            executable='server',
            name='softenable_display_server',
            output='screen'
        )
    )
    nodes_to_start.append(
        Node(
            package='softenable_display',
            executable='display_service',
            name='softenable_display_service',
            output='screen',
            parameters=[
                {'server_host': 'localhost'},
                {'server_port': LaunchConfiguration('port')},   # reuse the same port
                {'endpoint': '/update'},
            ],
            arguments=['--ros-args', '--log-level', 'rmw_cyclonedds_cpp:=error']  
        )
    )
    nodes_to_start.append(
        Node(
            package='stack_detect',
            executable='sam_service',
            name='sam_service',
            output="screen"
        ),
    )
    nodes_to_start.append(
        Node(
            package='gown_grasping',
            executable='grasp_dual_service',
            name='grasp_dual_service',
            output="screen"
        ),
    )
    nodes_to_start.append(
        Node(
            package='gown_opening',
            executable='gown_unfolding_service',
            name='gown_unfolding_service',
            output="screen"
        ),
    )
    return LaunchDescription(nodes_to_start)  