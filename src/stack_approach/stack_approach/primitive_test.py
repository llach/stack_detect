#!/usr/bin/env python3
import os
import sys
import glob
import time
import rclpy
import numpy as np

from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from control_msgs.action import FollowJointTrajectory
from softenable_display_msgs.srv import SetDisplay
from stack_msgs.srv import RollerGripper, StackDetect, MoveArm
from stack_approach.motion_helper import MotionHelper
from tf2_geometry_msgs import PointStamped, PoseStamped
from stack_approach.controller_switcher import ControllerSwitcher
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Point
from stack_approach.helpers import transform_to_pose_stamped, publish_img, point_to_pose, empty_pose, get_trafo, inv, matrix_to_pose_msg, pose_to_matrix, call_cli_sync
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from scipy.spatial.transform import Rotation as R


group2joints = {
    "both": [
        "left_arm_shoulder_pan_joint",
        "left_arm_shoulder_lift_joint",
        "left_arm_elbow_joint",
        "left_arm_wrist_1_joint",
        "left_arm_wrist_2_joint",
        "left_arm_wrist_3_joint",
        "right_arm_shoulder_pan_joint",
        "right_arm_shoulder_lift_joint",
        "right_arm_elbow_joint",
        "right_arm_wrist_1_joint",
        "right_arm_wrist_2_joint",
        "right_arm_wrist_3_joint"
    ],
    "left": [
        "left_arm_shoulder_pan_joint",
        "left_arm_shoulder_lift_joint",
        "left_arm_elbow_joint",
        "left_arm_wrist_1_joint",
        "left_arm_wrist_2_joint",
        "left_arm_wrist_3_joint"
    ],
    "right": [
        "right_arm_shoulder_pan_joint",
        "right_arm_shoulder_lift_joint",
        "right_arm_elbow_joint",
        "right_arm_wrist_1_joint",
        "right_arm_wrist_2_joint",
        "right_arm_wrist_3_joint"
    ]
}

group2controller = {
    "both": "dual_arm_joint_trajectory_controller",
    "left": "left_arm_joint_trajectory_controller",
    "right": "right_arm_joint_trajectory_controller",
}

def adjust_ts(ts, scaling=1, offset=0):
    return ((ts - ts[0]) * scaling ) + offset

class TrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('primitive_test')

        self.recbg = ReentrantCallbackGroup()

        self.controller_switcher = ControllerSwitcher()
        
        self.group2client = {
            "both": ActionClient(
                self, 
                FollowJointTrajectory, 
                f"/dual_arm_joint_trajectory_controller/follow_joint_trajectory",
                callback_group=self.recbg
            ),
            "left": ActionClient(
                self, 
                FollowJointTrajectory, 
                f"/left_arm_joint_trajectory_controller/follow_joint_trajectory",
                callback_group=self.recbg
            ),
            "right": ActionClient(
                self, 
                FollowJointTrajectory, 
                f"/right_arm_joint_trajectory_controller/follow_joint_trajectory",
                callback_group=self.recbg
            ),
        }

        self.finger2srv = {
            "left": self.create_client(RollerGripper, 'left_roller_gripper'),
            "right": self.create_client(RollerGripper, 'right_roller_gripper')
        }

        # for k, v in self.finger2srv.items():
        #     print(f"waiting for {k.upper()} gripper srv")
        #     while not v.wait_for_service(timeout_sec=2.0):
        #         self.get_logger().info('service not available, waiting again...')
        #     print(f"found {k.upper()} gripper srv")

        self.move_cli = self.create_client(MoveArm, "move_arm", callback_group=self.recbg)
        while not self.move_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('move arm service not available, waiting again...')


        self.finger_pose_pub = self.create_publisher(
            PoseStamped, "/finger_pose", 10, callback_group=self.recbg
        )

        self.wrist_pose_pub = self.create_publisher(
            PoseStamped, "/wrist_pose", 10, callback_group=self.recbg
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        print("waiting for tfs")
        while not (
            self.tf_buffer.can_transform("map", "right_finger", rclpy.time.Time()) and
            self.tf_buffer.can_transform("world", "right_arm_wrist_3_link", rclpy.time.Time())
        ):
            time.sleep(0.05)
            rclpy.spin_once(self)
        print("setup done!")

    def execute_traj(self, group, ts, qs):
        assert group in ["both", "left", "right"], f"unknown move group: {group}"

        print(f"executing trajectory with {len(qs)} points in {ts[-1]:.2f}s ...")
        
        self.controller_switcher.activate_controller(group2controller[group])

        traj = JointTrajectory()
        
        traj.points = []
        traj.joint_names = group2joints[group]

        for wp, t in zip(qs, ts):
            point = JointTrajectoryPoint()
            point.positions = list(wp)
            point.time_from_start.sec = int(t)
            point.time_from_start.nanosec = int((t % 1.0) * 1e9)
            traj.points.append(point)

        traj_goal = FollowJointTrajectory.Goal()
        traj_goal.trajectory = traj

        self.get_logger().info('{} waypoints [{} -> {}]'.format(len(traj.points), ts[0], ts[-1]))
        return self.group2client[group].send_goal_async(traj_goal)
    
    def call_cli_sync(self, cli, req):
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        return fut.result()
    
    def start_pose(self, time=5):
        fut = self.execute_traj(
            "right", 
            np.array([
                float(time),
            ]), 
            np.array([
                np.deg2rad([
                    -103.34,
                    -60.51,
                    122.96,
                    -143.36,
                    131.19,
                    209.58,
                ])
            ])
        )
        await_action_future(self, fut)

    def move_rel(self, x=0.0, y=0.0, z=0.0, time=5):
        msg = transform_to_pose_stamped(self.tf_buffer.lookup_transform("map", "right_arm_wrist_3_link", rclpy.time.Time()))

        msg.pose.position.x += x
        msg.pose.position.y += y
        msg.pose.position.z += z

        fut = self.move_cli.call_async(MoveArm.Request(
            target_pose = msg,
            execution_time = float(time),
            execute = True,
            controller_name = "right_arm_joint_trajectory_controller",
            ik_link = "right_arm_wrist_3_link",
            name_target = ["right_arm_shoulder_pan_joint", "right_arm_shoulder_lift_joint", "right_arm_elbow_joint", "right_arm_wrist_1_joint", "right_arm_wrist_2_joint", "right_arm_wrist_3_joint"]
        ))
        rclpy.spin_until_future_complete(self, fut)


def await_action_future(node, fut):
    print("waiting for future ...")
    while not fut.done() and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
    print("future done!")

    goal_handle = fut.result()
    if not goal_handle.accepted:
        print(f"Trajectory was rejected by the controller!")
        return False

    # now wait for result
    future_result = goal_handle.get_result_async()
    print("waiting for future result ...")
    while not future_result.done() and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
    print("future result done!")
    result = future_result.result()
    print(f"Trajectory finished with status: {result.status}")

    return True

def execute_opening(node, trajs):

    grasp_pose = empty_pose(frame="left_arm_wrist_3_link")

    fut = node.move_cli.call_async(MoveArm.Request(
        execute = True,
        target_pose = grasp_pose,
        execution_time = 0.7,
        controller_name = "left_arm_joint_trajectory_controller",
        ik_link = "left_arm_wrist_3_link",
        name_target = ["left_arm_shoulder_pan_joint", "left_arm_shoulder_lift_joint", "left_arm_elbow_joint", "left_arm_wrist_1_joint", "left_arm_wrist_2_joint", "left_arm_wrist_3_joint"]
    ))
    rclpy.spin_until_future_complete(node, fut)

    node.call_cli_sync(node.finger2srv["left"], RollerGripper.Request(roller_vel=80, roller_duration=5.0))
    node.call_cli_sync(node.finger2srv["left"], RollerGripper.Request(finger_pos=3500))


def main(args=None):
    GOAL_ANGLE = 40 # degrees
    TRANS_OFFSET_MAP = [0, 0.19, 0.03] # meters

    rclpy.init(args=args)
    node = TrajectoryPublisher()
    
    node.start_pose(time=2)

    Tmf = get_trafo("map", "right_finger", node.tf_buffer)
    Tfw = get_trafo("right_finger", "right_arm_wrist_3_link", node.tf_buffer)

    current_angle_offset = np.rad2deg(np.arccos(np.dot([0,1,0], Tmf[:3,:3]@[0,0,1]))) # angle between ground plane and z axis in wrist / finger frame
    print(f"map Y <-> finger Z {current_angle_offset:.2f}deg")

    goal_mat_finger = np.eye(4)
    goal_mat_finger[:3,:3] = goal_mat_finger[:3,:3] @ R.from_euler("xyz", [0, -(GOAL_ANGLE - current_angle_offset), 0], degrees=True).as_matrix()
    
    goal_mat_finger_in_map = Tmf @ goal_mat_finger
    goal_mat_finger_in_map[:3,3] += TRANS_OFFSET_MAP

    goal_mat_wrist_in_map = goal_mat_finger_in_map @ Tfw

    goal_pose_finger = matrix_to_pose_msg(goal_mat_finger_in_map, "map")
    goal_pose_wrist_in_finger  = matrix_to_pose_msg(goal_mat_wrist_in_map, "map")

    node.finger_pose_pub.publish(goal_pose_finger)
    node.wrist_pose_pub.publish(goal_pose_wrist_in_finger)

    fut = node.move_cli.call_async(MoveArm.Request(
        target_pose = goal_pose_wrist_in_finger,
        execution_time = 3.0,
        execute = True,
        controller_name = "right_arm_joint_trajectory_controller",
        ik_link = "right_arm_wrist_3_link",
        name_target = ["right_arm_shoulder_pan_joint", "right_arm_shoulder_lift_joint", "right_arm_elbow_joint", "right_arm_wrist_1_joint", "right_arm_wrist_2_joint", "right_arm_wrist_3_joint"]
    ))
    rclpy.spin_until_future_complete(node, fut)

    node.move_rel(y=0.03, time=1)
    node.move_rel(z=0.02, time=1)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
