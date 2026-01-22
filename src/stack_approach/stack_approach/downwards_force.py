#!/usr/bin/env python3
import time
import rclpy
import numpy as np

from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from control_msgs.action import FollowJointTrajectory
from stack_msgs.srv import MoveArm, RollerGripper
from tf2_geometry_msgs import PointStamped, PoseStamped
from stack_approach.controller_switcher import ControllerSwitcher
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from stack_approach.helpers import transform_to_pose_stamped, publish_img, point_to_pose, empty_pose, get_trafo, inv, matrix_to_pose_msg, pose_to_matrix, call_cli_sync
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import WrenchStamped

import sys
import select

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

class TrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('primitive_test')

        self.recbg = ReentrantCallbackGroup()
        self.controller_switcher = ControllerSwitcher()
  
        self.group2client = {
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

        # ---- F/T data ----
        self.latest_ft = None
        self.latest_ft_time = None
        self.ft_timeout = 0.1  # seconds

        self.create_subscription(
            WrenchStamped,
            '/ur5/ft_raw',
            self.ft_callback,
            10
        )

        self.move_cli = self.create_client(MoveArm, "move_arm", callback_group=self.recbg)
        while not self.move_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('move arm service not available, waiting again...')

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

    def get_fz(self):
        if self.latest_ft is None:
            return None
        return self.latest_ft.wrench.force.z

    def has_fresh_ft(self):
        if self.latest_ft is None or self.latest_ft_time is None:
            return False
        return (time.time() - self.latest_ft_time) < self.ft_timeout

    def ft_callback(self, msg: WrenchStamped):
        self.latest_ft = msg
        self.latest_ft_time = time.time()

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
    
    def go_to_q(self, q, time=3, side="left"):
        fut = self.execute_traj(
            side, 
            np.array([float(time)]), 
            np.array([q])
        )
        await_action_future(self, fut)

    def go_to_degrees(self, deg, time=3, side="left"):
        fut = self.execute_traj(
            side, 
            np.array([float(time)]), 
            np.array(np.deg2rad([deg]))
        )
        await_action_future(self, fut)

    def go_to_pose(self, pose, time, side="left"):
        fut = self.move_cli.call_async(MoveArm.Request(
            target_pose = pose,
            execution_time = float(time),
            execute = True,
            controller_name = f"{side}_arm_joint_trajectory_controller",
            ik_link = f"{side}_arm_wrist_3_link",
            name_target = [f"{side}_arm_shoulder_pan_joint", f"{side}_arm_shoulder_lift_joint", f"{side}_arm_elbow_joint", f"{side}_arm_wrist_1_joint", f"{side}_arm_wrist_2_joint", f"{side}_arm_wrist_3_joint"]
        ))
        rclpy.spin_until_future_complete(self, fut)

    def move_rel_wrist(self, x=0.0, y=0.0, z=0.0, time=5, side="left"):
        msg = transform_to_pose_stamped(self.tf_buffer.lookup_transform("map", f"{side}_arm_wrist_3_link", rclpy.time.Time()))

        msg.pose.position.x += x
        msg.pose.position.y += y
        msg.pose.position.z += z

        self.go_to_pose(msg, time)

    def ros_sleep(self, sec):
        for _ in range(int(sec/0.1)):
            time.sleep(0.1)
            rclpy.spin_once(self)

    def descend_until_force(
        self,
        force_goal: float,
        delta_z: float,
        traj_time: float,
        max_dist: float,
        side: str = "left",
    ):
        """
        Moves down in -Z until |Fz - Fz_ref| >= force_goal
        or max_dist is reached.

        Returns:
            True  -> force change detected
            False -> max distance reached without contact
        """

        # ----- Ensure fresh F/T data -----
        while rclpy.ok() and not self.has_fresh_ft():
            rclpy.spin_once(self, timeout_sec=0.1)

        fz_ref = self.get_fz()
        if fz_ref is None:
            self.get_logger().error("No initial Fz available")
            return False

        self.get_logger().info(
            f"Starting force descent, Fz_ref = {fz_ref:.2f} N"
        )

        steps = int(abs(max_dist / delta_z))
        moved_dist = 0.0

        for i in range(steps):

            if not self.has_fresh_ft():
                self.get_logger().warn("F/T data not fresh, waiting...")
                self.ros_sleep(0.05)
                continue

            fz = self.get_fz()
            if fz is None:
                continue

            # ----- Relative force condition -----
            if abs(fz - fz_ref) >= force_goal:
                self.get_logger().info(
                    f"Force change detected: "
                    f"|{fz:.2f} - {fz_ref:.2f}| >= {force_goal}"
                )
                return True

            # ----- Move down -----
            self.move_rel_wrist(
                z=-delta_z,
                time=traj_time,
                side=side
            )

            moved_dist += abs(delta_z)

        self.get_logger().info(
            "Max descent reached without detecting force change"
        )
        return False
    
def fc_down():
    node = TrajectoryPublisher()

    # ---- Go to start pose ----
    node.go_to_degrees(
        deg=[
            90.90,
            -116.09,
            -76.30,
            -265.4,
            44.87,
            -70.81
        ],
        time=2,
        side="left"
    )

    node.ros_sleep(1.0)

    # ---- Wait for fresh F/T data ----
    node.get_logger().info("Waiting for fresh F/T data...")
    while rclpy.ok() and not node.has_fresh_ft():
        rclpy.spin_once(node, timeout_sec=0.1)

    node.get_logger().info("F/T data ready, starting descent")

    # ---- Force-controlled descent ----
    success = node.descend_until_force(
        force_goal=5.0,     # N
        delta_z=0.0002,     # m
        traj_time=0.05,     # s
        max_dist=0.03,      # m
        side="left"
    )

    if success:
        node.get_logger().info("Contact detected successfully")
    else:
        node.get_logger().warn("Contact NOT detected")

    rclpy.shutdown()
  
if __name__ == '__main__':
    rclpy.init()
    fc_down()
