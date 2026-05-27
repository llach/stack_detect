from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, LaunchConfiguration, PathJoinSubstitution, TextSubstitution


from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


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
            executable='ft_publisher',
            name='ft_publisher',
            output="screen"
        ),
    )
    nodes_to_start.append(
        Node(
            package='iri_softenable_gripper',
            executable='softenable_gripper_node',
            name='right_softenable_gripper',
            remappings=[
            ('gripper_normalized', 'right_gripper_normalized'),
            ('roller_gripper', 'right_roller_gripper'),
            ],
            parameters=[PathJoinSubstitution([
                FindPackageShare('iri_softenable_gripper'),
                'config',
                'right.yaml'
                ])], # Uncomment this line to load param config_files
        ),
    )
    nodes_to_start.append(
        Node(
            package='iri_softenable_gripper',
            executable='softenable_gripper_node',
            name='left_softenable_gripper',
            remappings=[
            ('gripper_normalized', 'left_gripper_normalized'),
            ('roller_gripper', 'left_roller_gripper'),
            ],
            parameters=[PathJoinSubstitution([
                FindPackageShare('iri_softenable_gripper'),
                'config',
                'left.yaml'
                ])], # Uncomment this line to load param config_files
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
    return LaunchDescription(nodes_to_start)  