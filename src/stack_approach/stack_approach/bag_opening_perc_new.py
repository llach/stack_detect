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
from stack_msgs.srv import RollerGripper, StackDetect, MoveArm, RollerGripperV2
from stack_approach.motion_helper import MotionHelper
from stack_approach.controller_switcher import ControllerSwitcher
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Point
from stack_approach.helpers import empty_pose

from .motion_helper_v2 import group2joints, MotionHelperV2


class ServiceNode(Node):
    def __init__(self):
        super().__init__('bag_opening_perc')

        self.recbg = ReentrantCallbackGroup()

        # self.cli_display = self.create_client(SetDisplay, '/set_display')
        # while not self.cli_display.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('Waiting for /set_display service...')
    
        self.bag_cli = self.create_client(StackDetect, "detect_bag", callback_group=self.recbg)
        while not self.bag_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('bag detect service not available, waiting again...')


INITIAL_POSE = [ 1.39311028, -1.85827746, -2.0506866 , -0.98047812, -0.78614647, 0.45934969, -1.44216556, -0.68469267,  2.24536608, -3.02081587,  0.7967428 , -1.56835287 ] # bag in camera view

def adjust_ts(ts, scaling=1, offset=0):
    return ((ts - ts[0]) * scaling ) + offset

def build_traj(group, ts, qs):
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

    print('{} waypoints [{} -> {}]'.format(len(traj.points), ts[0], ts[-1]))


def load_trajectories(folder):
    files = glob.glob(os.path.join(folder, "t*.npz"))

    if not files:
        print("No files found.")
        return
    
    files = sorted(files, key=lambda f: int(os.path.splitext(os.path.basename(f))[0][1:]))

    print("Found files:", files)

    arrays = []
    for f in files:
        data = np.load(f)
        arr = dict(q=data["q"], ts=data["timestamps"])
        arrays.append(arr)

    return arrays


def move_over_bag(mh2: MotionHelperV2, left_wrist_pose):
    left_q = mh2.compute_ik_with_retries(left_wrist_pose, mh2.current_q.copy(), side="left")

    final_pose = np.concatenate(
        [
            left_q,
            np.deg2rad([
                -94.88,
                -51.86,
                135.98,
                -179.93,
                45.55,
                -72.64
            ])
        ]
    )

    mh2.go_to_q(final_pose, 10, side="both")

def execute_opening(mh2: MotionHelperV2, sn: ServiceNode, trajs):

    PRE_GRASP_HEIGHT = 0.837
    GRASP_HEIGHT = 0.790

    mh2.call_cli_sync(mh2.finger2srv["left_v2"], RollerGripperV2.Request(position=1.0))
    mh2.call_cli_sync(mh2.finger2srv["right_v2"], RollerGripperV2.Request(position=1.0))

    res = None
    while True:
        print("calling bag_cli")
        fut = sn.bag_cli.call_async(StackDetect.Request(
                offset = Point(
                    x=0.03, 
                    y=0.01,
                    z=PRE_GRASP_HEIGHT
                )
            )
        )
        rclpy.spin_until_future_complete(sn, fut)
        res = fut.result()

        inp = input("###################\n#######################\ngood?").lower().strip()
        if inp == "q":
            return
        elif inp == "y":
            break

    if not res:
        print("ERROR res is None")
        return
    
    bag_pose_wrist = res.target_pose
    
    move_over_bag(mh2, bag_pose_wrist)
    
    exit(0)
    # input("go down?")
    grasp_pose = empty_pose(frame="left_arm_wrist_3_link")
    grasp_pose.pose.position.z = PRE_GRASP_HEIGHT - GRASP_HEIGHT

    fut = node.move_cli.call_async(MoveArm.Request(
        execute = True,
        target_pose = grasp_pose,
        execution_time = 0.7,
        controller_name = "left_arm_joint_trajectory_controller",
        ik_link = "left_arm_wrist_3_link",
        name_target = ["left_arm_shoulder_pan_joint", "left_arm_shoulder_lift_joint", "left_arm_elbow_joint", "left_arm_wrist_1_joint", "left_arm_wrist_2_joint", "left_arm_wrist_3_joint"]
    ))
    rclpy.spin_until_future_complete(node, fut)

    node.call_cli_sync(node.finger2srv["left"], RollerGripper.Request(roller_vel=80, roller_duration=3.5))
    node.call_cli_sync(node.finger2srv["left_v2"], RollerGripperV2.Request(position=-1.0))

    grasp_pose_up = empty_pose(frame="left_arm_wrist_3_link")
    grasp_pose_up.pose.position.z = -0.02

    fut = node.move_cli.call_async(MoveArm.Request(
        execute = True,
        target_pose = grasp_pose_up,
        execution_time = 0.5,
        controller_name = "left_arm_joint_trajectory_controller",
        ik_link = "left_arm_wrist_3_link",
        name_target = ["left_arm_shoulder_pan_joint", "left_arm_shoulder_lift_joint", "left_arm_elbow_joint", "left_arm_wrist_1_joint", "left_arm_wrist_2_joint", "left_arm_wrist_3_joint"]
    ))
    rclpy.spin_until_future_complete(node, fut)

    # got to pre contact pose
    fut = node.execute_traj(
            "both", 
            np.array([
                2.5
            ]), 
            np.array([
                [1.56625366, -1.89681782, -2.23785257, -0.56585439,  0.59162313, -2.90597898,
                -1.76819355, -0.31465657,  2.04206449, -3.49937262,  0.81858134, -2.63544041]
            ])
        )
    await_action_future(node, fut)

    execute_trajectories(node, trajs)

def execute_trajectories(node, arrays):
    arrays.append({
        "ts": np.array([2.5]),
        "q": [
            np.deg2rad([
                74.16,
                -112.87,
                -94.81,
                -33.03,
                89.34,
                -183.47,
                -68.02,
                -47.74,
                62.89,
                -324.02,
                88.60,
                32.82,
            ])
        ]
    })
    
    offsets = [0 for _ in range(len(arrays))]
    offsets[0] = 10
    offsets[-1] = 2.5

    # offsets[2] = 10
    # offsets[4] = 10

    for i, arr in enumerate(arrays):
        if i < 4: continue

        print(f"--- T{i} ---")

        # speed up all movements slightly except downwards to table
        ts = adjust_ts(arr["ts"], offset = offsets[i], scaling=.7 if i == 2 else .7)

        ###### PRE ACTIONS
        if i == 5: # roll while rotating gripper
            node.finger2srv["right"].call_async(RollerGripper.Request(roller_vel=-80, roller_duration=ts[-1]*0.95))

        ##### TRAJECTORY EXEC
        fut = node.execute_traj(move_groups[i], ts, arr["q"])

        if fut:
           if not await_action_future(node, fut): break
        
        ###### POST ACTIONS
        if i == 2: # grasp bag
            node.call_cli_sync(node.finger2srv["left"], RollerGripper.Request(roller_vel=80, roller_duration=5.0))
            node.call_cli_sync(node.finger2srv["left"], RollerGripper.Request(finger_pos=3500))
            # node.cli_display.call_async(SetDisplay.Request(name="protocol_9"))
        elif i == 5: # close right gripper
            node.call_cli_sync(node.finger2srv["right"], RollerGripper.Request(finger_pos=700))

def main(args=None):
    arrays = load_trajectories(f"/home/ros/ws/src/bag_opening/trajectories/")
    # arrays = load_trajectories(f"{os.environ['HOME']}/projects/se_clinic_case/ros_modules/bag_opening/trajectories/")

    rclpy.init(args=args)
    mh2 = MotionHelperV2()
    sn = ServiceNode()
    
    last_arg = sys.argv[-1]

    input("start?")
    mh2.go_to_q(INITIAL_POSE, 5, "both")

    if "bag_opening_perc" in last_arg: # with "in", we catch execution with python and ros2 run
        print("executing normally ...")
        execute_opening(mh2, sn, arrays)

    # elif last_arg == "slides":
    #     print("executing with slides ...")
    #     # node.cli_display.call_async(SetDisplay.Request(name="protocol_1"))
    #     # time.sleep(10)
    #     node.cli_display.call_async(SetDisplay.Request(name="protocol_bag_1"))

    #     execute_opening(node, arrays)
    #     node.cli_display.call_async(SetDisplay.Request(name="protocol_bag_2"))
    #     time.sleep(7)
    #     node.cli_display.call_async(SetDisplay.Request(name="protocol_11"))


    mh2.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
