#!/usr/bin/env python3
import os
import glob
import rclpy
import numpy as np

from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from control_msgs.action import FollowJointTrajectory
from stack_msgs.srv import RollerGripper
from stack_approach.motion_helper import MotionHelper
from stack_approach.controller_switcher import ControllerSwitcher
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


move_groups = [
    "both",
    "both",
    "left",
    "left",
    "right",
    "right",
    "both",
    "both",
]

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
        super().__init__('send_joint_trajectory')

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

        for k, v in self.finger2srv.items():
            print(f"waiting for {k.upper()} gripper srv")
            while not v.wait_for_service(timeout_sec=2.0):
                self.get_logger().info('service not available, waiting again...')

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

def await_action_future(node, fut):
    rclpy.spin_until_future_complete(node, fut)

    goal_handle = fut.result()
    if not goal_handle.accepted:
        print(f"Trajectory was rejected by the controller!")
        return False

    # now wait for result
    future_result = goal_handle.get_result_async()
    rclpy.spin_until_future_complete(node, future_result)
    result = future_result.result()
    print(f"Trajectory finished with status: {result.status}")

    return True

def main(args=None):
    arrays = load_trajectories(f"/home/ros/ws/src/bag_opening/trajectories/")
    
    rclpy.init(args=args)
    node = TrajectoryPublisher()
 
    # Spin once or twice to allow action clients to connect
    for _ in range(5):
        rclpy.spin_once(node, timeout_sec=0.1)

    offsets = [0 for _ in range(len(arrays))]
    offsets[0] = 10

    # offsets[2] = 10
    # offsets[4] = 10

    for i, arr in enumerate(arrays):
        print(f"--- T{i} ---")

        # speed up all movements slightly except downwards to table
        ts = adjust_ts(arr["ts"], offset = offsets[i], scaling=1 if i == 2 else .65)

        ###### PRE ACTIONS
        if i == 0:
            node.call_cli_sync(node.finger2srv["right"], RollerGripper.Request(finger_pos=1650))
            node.call_cli_sync(node.finger2srv["left"], RollerGripper.Request(finger_pos=2950))
        elif i == 5:
            node.finger2srv["right"].call_async(RollerGripper.Request(roller_vel=-80, roller_duration=ts[-1]*0.95))

        ##### TRAJECTORY EXEC
        fut = node.execute_traj(move_groups[i], ts, arr["q"])

        if fut:
           if not await_action_future(node, fut): break
        
        ###### POST ACTIONS
        if i == 2:
            node.call_cli_sync(node.finger2srv["left"], RollerGripper.Request(roller_vel=80, roller_duration=5.0))
            node.call_cli_sync(node.finger2srv["left"], RollerGripper.Request(finger_pos=3450))
        elif i == 5:
            node.call_cli_sync(node.finger2srv["right"], RollerGripper.Request(finger_pos=850))
        # elif i == 7:
        #     node.call_cli_sync(node.finger2srv["right"], RollerGripper.Request(finger_pos=1900))
        #     node.call_cli_sync(node.finger2srv["left"], RollerGripper.Request(finger_pos=2000))

        # if i == 0: break
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
