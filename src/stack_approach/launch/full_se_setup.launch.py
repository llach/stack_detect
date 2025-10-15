from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():

    nodes_to_start = []
    # TODO NEED TO INCLUDE THE LAUNCH FILES FOR:
    # Start the wrist camera
    # LD_LIBRARY_PATH=/opt/conda/envs/softenable/lib/:$LD_LIBRARY_PATH ros2 launch realsense2_camera rs_launch.py config_file:=/home/ros/ws/src/iri_ur5e_description/config/l515.yaml

    # # Start the camera for unfolding/YOLO
    # LD_LIBRARY_PATH=/opt/conda/envs/softenable/lib/:$LD_LIBRARY_PATH ros2 launch realsense2_camera rs_launch.py config_file:=/home/ros/ws/src/iri_ur5e_description/config/d435.yaml camera_name:=unfolding_camera

    # # Start the UR5 controllers
    # ros2 launch iri_ur5e_description ur_dual.launch.py

    # # Start the MoveIt package for the IK solver
    # ros2 launch dual_ur5e_moveit move_group.launch.py

    # # Start the set-up for the stack approach (including grippers)
    # ros2 launch stack_approach se_setup.launch.py

    realsense2_camera_ros = FindPackageShare(package='realsense2_camera').find('realsense2_camera')  
    iri_ur5_description_ros = FindPackageShare(package='iri_ur5e_description').find('iri_ur5e_description')  
    dual_ur5e_moveit = FindPackageShare(package='dual_ur5e_moveit').find('dual_ur5e_moveit') 
    stack_approach = FindPackageShare(package='stack_approach').find('stack_approach') 

    # Cameras
    nodes_to_start.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource([realsense2_camera_ros, '/launch/rs_launch.py']),
        launch_arguments={'config_file': "/home/ros/ws/src/iri_ur5e_description/config/l515.yaml"
                          }.items()))
    nodes_to_start.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource([realsense2_camera_ros, '/launch/rs_launch.py']),
        launch_arguments={'config_file': "/home/ros/ws/src/iri_ur5e_description/config/d435.yaml",
                          'camera_name': 'unfolding_camera'
                          }.items()))
    # UR5e control
    nodes_to_start.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource([iri_ur5_description_ros, '/launch/ur_dual.py'])))
    # Moveit 
    nodes_to_start.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource([dual_ur5e_moveit, '/launch/move_group.py'])))
    # Stack approach setup
    nodes_to_start.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource([stack_approach, '/launch/se_setup.py'])))
    

    ## Nodes
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