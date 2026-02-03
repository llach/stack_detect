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

from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest, RobotState
from sensor_msgs.msg import JointState

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

        self.ik_client = self.create_client(
            srv_type=GetPositionIK,
            srv_name="compute_ik",
            callback_group=self.recbg,
        )

        # ---- F/T data ----
        self.latest_ft = None
        self.latest_ft_time = None
        self.data_timeout = 0.1  # seconds

        self.create_subscription(
            WrenchStamped,
            '/ur5/ft_raw',
            self.ft_callback,
            10
        )

        self.current_q = None
        self.latest_js_time = None
        self.create_subscription(
            JointState, "/joint_states", self.js_callback, 0#, callback_group=self.recbg
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

    def js_callback(self, msg):
        self.current_q = {jname: q for jname, q in zip(msg.name, msg.position)}
        self.latest_js_time = time.time()

    def get_fz(self):
        if self.latest_ft is None:
            return None
        return self.latest_ft.wrench.force.z
    
    def get_f(self):
        if self.latest_ft is None:
            return None
        return [
            self.latest_ft.wrench.force.x,
            self.latest_ft.wrench.force.y,
            self.latest_ft.wrench.force.z
        ]

    def has_fresh_ft(self):
        if self.latest_ft is None or self.latest_ft_time is None:
            return False
        return (time.time() - self.latest_ft_time) < self.data_timeout
    
    def has_fresh_js(self):
        if self.current_q is None or self.latest_js_time is None:
            return False
        return (time.time() - self.latest_js_time) < self.data_timeout
    
    def get_js(self):
        if self.current_q is None: return None
        return self.current_q.copy()

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

    def go_to_pose(self, pose, time, side="left", blocking=True):
        fut = self.move_cli.call_async(MoveArm.Request(
            target_pose = pose,
            execution_time = float(time),
            execute = True,
            controller_name = f"{side}_arm_joint_trajectory_controller",
            ik_link = f"{side}_arm_wrist_3_link",
            name_target = [f"{side}_arm_shoulder_pan_joint", f"{side}_arm_shoulder_lift_joint", f"{side}_arm_elbow_joint", f"{side}_arm_wrist_1_joint", f"{side}_arm_wrist_2_joint", f"{side}_arm_wrist_3_joint"]
        ))
        if not blocking: return fut
        rclpy.spin_until_future_complete(self, fut)

    def move_rel_wrist(self, x=0.0, y=0.0, z=0.0, time=5, side="left", blocking=True):
        msg = transform_to_pose_stamped(self.tf_buffer.lookup_transform("map", f"{side}_arm_wrist_3_link", rclpy.time.Time()))

        msg.pose.position.x += x
        msg.pose.position.y += y
        msg.pose.position.z += z

        return self.go_to_pose(msg, time, side=side, blocking=blocking)
    
    def compute_ik_with_retries(
        self,
        pose_stamped: PoseStamped,
        current_state: dict,
        side: str,
        retries: int = 3,
    ):
        joint_names = list(current_state.keys())
        qs = list(current_state.values())

        for attempt in range(retries):
            self.get_logger().info(f"MoveIt IK attempt {attempt+1}/{retries}")

            req = GetPositionIK.Request()
            ik_req = PositionIKRequest()

            ik_req.group_name = f"{side}_arm"
            ik_req.pose_stamped = pose_stamped
            ik_req.ik_link_name = f"{side}_arm_wrist_3_link"
            ik_req.robot_state.joint_state.name = joint_names
            ik_req.robot_state.joint_state.position = qs
            ik_req.timeout.sec = 0
            ik_req.timeout.nanosec = int(0.2 * 1e9)
            ik_req.avoid_collisions = False  # set True if you want collision checking

            req.ik_request = ik_req

            fut = self.ik_client.call_async(req)
            rclpy.spin_until_future_complete(self, fut)
            res = fut.result()

            if res is None:
                self.get_logger().warn("IK service returned None")
                continue

            if res.error_code.val != res.error_code.SUCCESS:
                self.get_logger().warn(f"IK failed with code {res.error_code.val}")
                continue

            sol_state = res.solution.joint_state
            print("ss", sol_state)

            # Map solution joints into controller joint order
            sol_map = dict(zip(sol_state.name, sol_state.position))
            try:
                q = [sol_map[j] for j in joint_names]
            except KeyError as e:
                self.get_logger().error(f"IK solution missing joint {e}")
                continue

            return q

        self.get_logger().error("MoveIt IK failed after retries")
        return None


    def ros_sleep(self, sec):
        for _ in range(int(sec/0.1)):
            time.sleep(0.1)
            rclpy.spin_once(self)

    def descend_until_force_step(
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
    
    def move_along_z_until_force_traj(
        self,
        force_goal: float,
        traj_time: float,
        dist: float,
        side: str = "left",
    ):

        while rclpy.ok() and not self.has_fresh_ft() and not self.has_fresh_js():
            rclpy.spin_once(self, timeout_sec=0.1)

        fz_ref = self.get_fz()
        if fz_ref is None:
            self.get_logger().error("No initial Fz available")
            return False

        self.get_logger().info(f"Starting force descent, Fz_ref = {fz_ref:.2f} N")


        traj = self.build_cartesian_down_traj(dist, traj_time, side)
        if traj is None:
            self.get_logger().error("Failed to build descent trajectory")
            return False

        goal_handle = self.send_traj_and_get_handle(traj, side)
        if goal_handle is None:
            self.get_logger().error("Failed to execute descent trajectory")
            return False

        result_future = goal_handle.get_result_async()

        # ---- Monitor force while moving ----
        while rclpy.ok() and not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.01)

            if not self.has_fresh_ft():
                continue

            fz = self.get_fz()
            if fz is None:
                continue

            if abs(fz - fz_ref) >= force_goal:
                self.get_logger().info(
                    f"Force change detected: |{fz:.2f} - {fz_ref:.2f}| >= {force_goal}"
                )
                self.get_logger().warn("Cancelling trajectory due to contact!")

                cancel_fut = goal_handle.cancel_goal_async()
                rclpy.spin_until_future_complete(self, cancel_fut)
                return True

        self.get_logger().info("Descent finished without detecting force change")
        return False

    def build_cartesian_down_traj(self, dist: float, traj_time: float, side: str):
        # Current EE pose in map frame
        tf = self.tf_buffer.lookup_transform(
            "map",
            f"{side}_arm_wrist_3_link",
            rclpy.time.Time()
        )

        start_pose = transform_to_pose_stamped(tf)

        target_pose = PoseStamped()
        target_pose.header.frame_id = "map"
        target_pose.pose = start_pose.pose
        target_pose.pose.position.z += dist

        js = self.get_js()
        current_state = {jn: js[jn] for jn in group2joints[side]}
        q_target = self.compute_ik_with_retries(target_pose, current_state, side)
        if q_target is None:
            return None

        # We assume controller starts from current state, so single waypoint is fine
        traj = JointTrajectory()
        traj.joint_names = group2joints[side]

        current_q = self.get_js()
        print([current_q[jn] for jn in group2joints[side]])
        print(q_target)

        point = JointTrajectoryPoint()
        point.positions = list(q_target)
        point.time_from_start.sec = int(traj_time)
        point.time_from_start.nanosec = int((traj_time % 1.0) * 1e9)

        traj.points.append(point)
        return traj

    def send_traj_and_get_handle(self, traj: JointTrajectory, side: str):
        self.controller_switcher.activate_controller(group2controller[side])

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        client = self.group2client[side]
        fut = client.send_goal_async(goal)

        while not fut.done() and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)

        goal_handle = fut.result()

        if not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected")
            return None

        self.get_logger().info("Trajectory goal accepted")
        return goal_handle


    def open_gripper_on_force_change(self, threshold: float, sleep_time: float = 0.05):
        """
        Monitors F/T data and opens gripper if the sum of absolute
        differences from the initial force vector exceeds `threshold`.

        Args:
            threshold: float, N, sum of |Fx-Fx0| + |Fy-Fy0| + |Fz-Fz0|
            sleep_time: float, seconds between iterations

        Returns:
            True  -> gripper was opened
            False -> exited loop without opening
        """

        # ----- Ensure fresh F/T data -----
        while rclpy.ok() and not self.has_fresh_ft():
            rclpy.spin_once(self, timeout_sec=0.1)

        f_ref = self.get_f()
        if f_ref is None:
            self.get_logger().error("No F/T data available for reference")
            return False

        self.get_logger().info(f"Starting force monitor, ref={f_ref}")

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)

            f_cur = self.get_f()
            if f_cur is None:
                continue

            # sum of absolute differences
            diff_sum = sum(abs(fc - fr) for fc, fr in zip(f_cur, f_ref))

            if diff_sum >= threshold:
                self.get_logger().info(
                    f"Force threshold exceeded: sum(|ΔF|)={diff_sum:.2f} >= {threshold}"
                )
                # Open gripper (replace with your gripper call)
                node.call_cli_sync(node.finger2srv["left"], RollerGripper.Request(finger_pos=2000))
                return True

            # optional small sleep to throttle loop
            self.ros_sleep(sleep_time)

        return False


def force_open(node: TrajectoryPublisher, force_lim = 4):
    node.call_cli_sync(node.finger2srv["left"], RollerGripper.Request(finger_pos=2000))
    # node.call_cli_sync(node.finger2srv["left"], RollerGripper.Request(roller_vel=80, roller_duration=5.0))
    input("cont?")
    node.call_cli_sync(node.finger2srv["left"], RollerGripper.Request(finger_pos=3300))
    node.open_gripper_on_force_change(force_lim)

def fc_down(node, force_goal=5):
    start_pose = [
        85.54,
        -117.26,
        -68.25,
        -278.44,
        46.06,
        -61.97
    ]

    # ---- Go to start pose ----
    node.go_to_degrees(
        deg=start_pose,
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
    # success = node.descend_until_force(
    #     force_goal=force_goal,     # N
    #     delta_z=0.0004,     # m
    #     traj_time=0.02,     # s
    #     max_dist=0.03,      # m
    #     side="left"
    # )

    success = node.move_along_z_until_force_traj(
        force_goal=force_goal,     # N
        traj_time=5.0,     # s
        dist=-0.02,      # m
        side="left"
    )

    if success:
        node.get_logger().info("Contact detected successfully")
    else:
        node.get_logger().warn("Contact NOT detected")

    input("cont?")
    # ---- Go to start pose ----
    node.go_to_degrees(
        deg=start_pose,
        time=2,
        side="left"
    )

    rclpy.shutdown()
  
if __name__ == '__main__':
    rclpy.init()
    node = TrajectoryPublisher()

    fd = True

    if fd:
        fc_down(node, force_goal=4)
    else:
        force_open(node, force_lim=6)
