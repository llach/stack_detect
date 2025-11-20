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
        super().__init__('bag_opening')

        self.recbg = ReentrantCallbackGroup()

        self.controller_switcher = ControllerSwitcher()

        self.cli_display = self.create_client(SetDisplay, '/set_display')
        while not self.cli_display.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /set_display service...')
        
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
    
    """
    [np.concatenate([
                np.deg2rad([ 
                    -82.63, 
                    -39.23, 
                    128.65, 
                    -173.08, 
                    45.65, 
                    -89.86,
                ]), 
                [
                    -1.63986427, 
                    -0.50394781,  
                    2.1300605 , 
                    -3.27673878, 
                    0.80105019, 
                    -2.82983905
                ]
            ])]
    """

    def initial_pose(self):
        fut = self.execute_traj(
            "both", 
            np.array([4]), 
            np.array([
                [ 1.39311028, -1.85827746, -2.0506866 , -0.98047812, -0.78614647, 0.45934969, -1.63986427, -0.50394781,  2.1300605 , -3.27673878, 0.80105019, -2.82983905] # old  start (cam facing front)
            ])
        )
        await_action_future(self, fut)

    def initial_pose_new(self):
        fut = self.execute_traj(
            "both", 
            np.array([4]), 
            np.array([
                [  1.39311028, -1.85827746, -2.0506866 , -0.98047812, -0.78614647, 0.45934969, -1.44216556, -0.68469267,  2.24536608, -3.02081587,  0.7967428 , -1.56835287 ] # bag in camera view
            ])
        )
        await_action_future(self, fut)

    def retreat(self):
        fut = self.execute_traj(
            "both", 
            np.array([
                3,
                5,
                # 30
            ]), 
            np.array([
                # np.deg2rad([        # start pose (offering)
                #     74.16,
                #     -112.87,
                #     -94.81,
                #     -33.03,
                #     89.34,
                #     -183.47,

                #     -68.02,
                #     -47.74,
                #     62.89,
                #     -324.02,
                #     88.60,
                #     32.82,
                # ]),
                np.deg2rad([        # intermediate
                    94.97,
                    -100.60,
                    -106.19,
                    -40.79,
                    6.58,
                    -79.47,

                    -91.69,
                    -38.37,
                    89.48,
                    -165.55,
                    38.88,
                    139.85,
                ]),
                np.deg2rad([        # unstacking start right / unfolding home left
                    108.67,
                    -92.52,
                    -113.68,
                    -45.90,
                    -47.90,
                    -11.00,

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

def execute_opening(node, arrays):
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

    node.call_cli_sync(node.finger2srv["right"], RollerGripper.Request(finger_pos=1650))
    node.call_cli_sync(node.finger2srv["left"], RollerGripper.Request(finger_pos=2950))

    for i, arr in enumerate(arrays):
        if i == 0: continue

        print(f"--- T{i} ---")

        # speed up all movements slightly except downwards to table
        ts = adjust_ts(arr["ts"], offset = offsets[i], scaling=.7 if i == 2 else .7)

        ###### PRE ACTIONS
        if i == 5:
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

def calib(node, arrays):
    # print("going to initial pose ...")
    fut = node.execute_traj("both", [7], arrays[0]["q"])
    await_action_future(node, fut)

    fut = node.execute_traj("both", adjust_ts(arrays[1]["ts"], scaling=0.5), arrays[1]["q"])
    await_action_future(node, fut)

    while True:
        fut = node.execute_traj("left", adjust_ts(arrays[2]["ts"], scaling=1.3), arrays[2]["q"])
        await_action_future(node, fut)

        time.sleep(1)

        fut = node.execute_traj("left", [2], [arrays[2]["q"][0]])
        await_action_future(node, fut)

        if input("continue?").lower() == "q":
            print("bye!")
            break

def main(args=None):
    arrays = load_trajectories(f"/home/ros/ws/src/bag_opening/trajectories/")
    # arrays = load_trajectories(f"{os.environ['HOME']}/projects/se_clinic_case/ros_modules/bag_opening/trajectories/")

    rclpy.init(args=args)
    node = TrajectoryPublisher()
    
    last_arg = sys.argv[-1]

    if last_arg == "initial":
        print("going to intial pose ...")
        node.cli_display.call_async(SetDisplay.Request(name="protocol_1", use_tts=False))
        node.initial_pose()
    elif last_arg == "initial_new":
        print("going to initial_new pose ...")
        # node.cli_display.call_async(SetDisplay.Request(name="protocol_1", use_tts=False))
        node.initial_pose_new()
    elif last_arg == "retreat":
        print("going to retreat pose ...")
        # node.cli_display.call_async(SetDisplay.Request(name="protocol_2"))
        node.retreat()
    elif last_arg == "calib":
        print("doing bag calib")
        calib(node, arrays)
    elif last_arg == "slides":
        print("executing with slides ...")
        node.cli_display.call_async(SetDisplay.Request(name="protocol_1"))
        time.sleep(10)
        node.cli_display.call_async(SetDisplay.Request(name="protocol_bag_1"))

        execute_opening(node, arrays)
        node.cli_display.call_async(SetDisplay.Request(name="protocol_bag_2"))
    else:
        print("opening bag ...")
        execute_opening(node, arrays)
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
