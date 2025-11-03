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
from stack_approach.controller_switcher import ControllerSwitcher
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Point
from stack_approach.helpers import empty_pose


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
        super().__init__('send_joint_trajectory')

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

        for k, v in self.finger2srv.items():
            print(f"waiting for {k.upper()} gripper srv")
            while not v.wait_for_service(timeout_sec=2.0):
                self.get_logger().info('service not available, waiting again...')
            print(f"found {k.upper()} gripper srv")


        self.move_cli = self.create_client(MoveArm, "move_arm", callback_group=self.recbg)
        while not self.move_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('move arm service not available, waiting again...')


        self.bag_cli = self.create_client(StackDetect, "detect_bag", callback_group=self.recbg)
        while not self.bag_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('bag detect service not available, waiting again...')


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

def move_over_bag(node, left_wrist_pose):
    mar_left_pre = MoveArm.Request(
        target_pose = left_wrist_pose,
        execute = False,
        controller_name = "left_arm_joint_trajectory_controller",
        execution_time = 2.5,
        ik_link = "left_arm_wrist_3_link",
        name_target = ["left_arm_shoulder_pan_joint", "left_arm_shoulder_lift_joint", "left_arm_elbow_joint", "left_arm_wrist_1_joint", "left_arm_wrist_2_joint", "left_arm_wrist_3_joint"]
    )
    fut = node.move_cli.call_async(mar_left_pre)
    rclpy.spin_until_future_complete(node, fut)

    final_pose = np.concatenate(
        [
            fut.result().q_end,
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

    print("over bag pose", final_pose)

    fut = node.execute_traj(
            "both", 
            np.array([
                3.0
            ]), 
            np.array([
                final_pose
            ])
        )
    await_action_future(node, fut)

def execute_opening(node, trajs):

    PRE_GRASP_HEIGHT = 0.835
    GRASP_HEIGHT = 0.7905

    node.call_cli_sync(node.finger2srv["right"], RollerGripper.Request(finger_pos=1650))
    node.call_cli_sync(node.finger2srv["left"], RollerGripper.Request(finger_pos=2950))

    res = None
    while True:
        print("calling bag_cli")
        fut = node.bag_cli.call_async(StackDetect.Request(
                offset = Point(
                    x=0.035, 
                    y=-0.025, 
                    z=PRE_GRASP_HEIGHT
                )
            )
        )
        rclpy.spin_until_future_complete(node, fut)
        res = fut.result()

        inp = input("good?").lower().strip()
        if inp == "q":
            return
        elif inp == "y":
            break

    if not res:
        print("ERROR res is None")
        return
    
    bag_pose_wrist = res.target_pose
    
    move_over_bag(node, bag_pose_wrist)

    input("go down?")
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

    node.call_cli_sync(node.finger2srv["left"], RollerGripper.Request(roller_vel=80, roller_duration=5.0))
    node.call_cli_sync(node.finger2srv["left"], RollerGripper.Request(finger_pos=3500))

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
            node.call_cli_sync(node.finger2srv["left"], RollerGripper.Request(finger_pos=3450))
        elif i == 5: # close right gripper
            node.call_cli_sync(node.finger2srv["right"], RollerGripper.Request(finger_pos=700))

def main(args=None):
    arrays = load_trajectories(f"/home/ros/ws/src/bag_opening/trajectories/")
    # arrays = load_trajectories(f"{os.environ['HOME']}/projects/se_clinic_case/ros_modules/bag_opening/trajectories/")

    rclpy.init(args=args)
    node = TrajectoryPublisher()
    
    last_arg = sys.argv[-1]

    if "bag_opening_perc" in last_arg: # with "in", we catch execution with python and ros2 run
        execute_opening(node, arrays)
    elif last_arg == "slides":
        print("executing with slides ...")
        node.cli_display.call_async(SetDisplay.Request(name="protocol_1"))
        time.sleep(10)
        node.cli_display.call_async(SetDisplay.Request(name="protocol_bag_1"))

        execute_opening(node, arrays)
        node.cli_display.call_async(SetDisplay.Request(name="protocol_bag_2"))


    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
