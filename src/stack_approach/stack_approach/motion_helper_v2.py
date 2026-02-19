#!/usr/bin/env python3
import time
import rclpy
import numpy as np

from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from control_msgs.action import FollowJointTrajectory
from stack_msgs.srv import MoveArm, RollerGripper, RollerGripperV2
from tf2_geometry_msgs import PoseStamped
from stack_approach.controller_switcher import ControllerSwitcher
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from stack_approach.helpers import transform_to_pose_stamped
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import WrenchStamped

from moveit_msgs.srv import GetMotionPlan, GetCartesianPath
from moveit_msgs.msg import MotionPlanRequest, Constraints, JointConstraint, MoveItErrorCodes
from trajectory_msgs.msg import JointTrajectory
from moveit_msgs.msg import DisplayTrajectory


from scipy.spatial.transform import Rotation as R

from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest, RobotState
from sensor_msgs.msg import JointState

def await_traj_goal_handle(node, goal_handle):
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

class MotionHelperV2(Node):
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
            "both": ActionClient(
                self, 
                FollowJointTrajectory, 
                f"/dual_arm_joint_trajectory_controller/follow_joint_trajectory",
                callback_group=self.recbg
            ),
        }

        self.finger2srv = {
            "left": self.create_client(RollerGripper, 'left_roller_gripper'),
            "right": self.create_client(RollerGripper, 'right_roller_gripper'),
            "left_v2": self.create_client(RollerGripperV2, 'left_gripper_normalized'),
            "right_v2": self.create_client(RollerGripperV2, 'right_gripper_normalized')
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

        self.plan_client = self.create_client(GetMotionPlan, '/plan_kinematic_path')
        while not self.plan_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /plan_kinematic_path service...')
 
        # Publisher for RViz visualization
        self.traj_pub = self.create_publisher(DisplayTrajectory, '/display_planned_path2', 10)

        self.cartesian_path_client = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        while not self.cartesian_path_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /compute_cartesian_path...')
 
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

    def call_cli_sync(self, cli, req):
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        return fut.result()

    def execute_trajectory(self, traj, side):
        gh = self.send_traj_and_get_handle(traj, side=side)
        await_traj_goal_handle(self, gh)

    def go_to_q(self, q, time=3, side="left"):
        traj = self.create_singe_point_traj(
            names=group2joints[side],
            q=q,
            time=time
        )
        print("sending")
        gh = self.send_traj_and_get_handle(traj, side=side)
        await_traj_goal_handle(self, gh)

    def go_to_degrees(self, deg, time=3, side="left"):
        traj = self.create_singe_point_traj(
            names=group2joints[side],
            q=np.deg2rad(deg),
            time=time
        )
        gh = self.send_traj_and_get_handle(traj, side=side)
        await_traj_goal_handle(self, gh)

    def go_to_pose(self, pose, time, side="left", blocking=True):
        traj = self.get_traj(pose, side=side, traj_time=time)
        gh = self.send_traj_and_get_handle(traj, side=side)
   
        if not blocking: return gh
        await_traj_goal_handle(self, gh)

    def move_rel_wrist(self, x=0.0, y=0.0, z=0.0, time=5, side="left", blocking=True):
        msg = transform_to_pose_stamped(self.tf_buffer.lookup_transform("map", f"{side}_arm_wrist_3_link", rclpy.time.Time()))

        msg.pose.position.x += x
        msg.pose.position.y += y
        msg.pose.position.z += z

        return self.go_to_pose(msg, time, side=side, blocking=blocking)
    
    def get_tf_as_pose(self, fro, to):
        return transform_to_pose_stamped(self.tf_buffer.lookup_transform(fro, to, rclpy.time.Time()))

    def move_rel_finger(self, x=0.0, y=0.0, z=0.0, time=5, side="left", blocking=True):
        # 1. Lookup the current transform from map to the finger
        t = self.tf_buffer.lookup_transform("map", f"{side}_finger", rclpy.time.Time())

        # 2. Apply translation in the local (child) frame
        # We add the offsets to the existing translation relative to the current orientation
        # To do this correctly in 3D space, we treat the (x, y, z) as a vector 
        # and rotate it by the finger's current orientation.
        
        # Get the current orientation as a rotation object
        quat = [t.transform.rotation.x, t.transform.rotation.y, 
                t.transform.rotation.z, t.transform.rotation.w]
        rotation = R.from_quat(quat)

        # Rotate the local movement vector into the map frame
        local_move = np.array([x, y, z])
        map_move = rotation.apply(local_move)

        # 3. Apply the rotated movement to the map-frame position
        t.transform.translation.x += map_move[0]
        t.transform.translation.y += map_move[1]
        t.transform.translation.z += map_move[2]

        # 4. Convert the updated transform to a PoseStamped message
        msg = transform_to_pose_stamped(t)

        # transform to wrist frame
        msg = self.finger_to_wrist(msg, side=side)

        return self.go_to_pose(msg, time, side=side, blocking=blocking)
    
    def create_singe_point_traj(self, names, q, time):
        return JointTrajectory(
            joint_names=names,
            points=[
                JointTrajectoryPoint(
                    positions=list(q),
                    time_from_start=rclpy.duration.Duration(seconds=time).to_msg()
                )
            ]
        )

    def compute_ik_with_retries(
        self,
        pose_stamped: PoseStamped,
        current_state: dict,
        side: str,
        retries: int = 3,
    ):
        
        joint_names = []
        qs = []
        for jn in group2joints[side]:
            joint_names.append(jn)
            qs.append(current_state[jn])

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

            print(ik_req)

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
                q = [sol_map[j] for j in joint_names if j in group2joints[side]]
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
    

    def finger_to_wrist(self, finger_pose_map, side="left"):
        """
        Transforms a pose representing the finger's target in the map frame 
        into the corresponding wrist pose in the map frame.
        """
        # 1. Get the transform from the finger to the wrist
        # We need to know where the wrist is relative to the finger
        try:
            # lookup_transform(target_frame, source_frame, time)
            finger_to_wrist_tf = self.tf_buffer.lookup_transform(
                f"{side}_finger", 
                f"{side}_arm_wrist_3_link", 
                rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().error(f"Could not lookup transform: {e}")
            return None

        # 2. Convert finger_pose_map to a matrix or use scipy for the math
        # Finger orientation in map
        f_quat = [finger_pose_map.pose.orientation.x, finger_pose_map.pose.orientation.y, 
                finger_pose_map.pose.orientation.z, finger_pose_map.pose.orientation.w]
        f_rot = R.from_quat(f_quat)
        
        # Offset (wrist relative to finger)
        offset_vec = np.array([
            finger_to_wrist_tf.transform.translation.x,
            finger_to_wrist_tf.transform.translation.y,
            finger_to_wrist_tf.transform.translation.z
        ])

        # 3. Rotate the offset by the finger's map orientation and add to map position
        # This finds the wrist position in the map frame
        wrist_pos_map = np.array([
            finger_pose_map.pose.position.x,
            finger_pose_map.pose.position.y,
            finger_pose_map.pose.position.z
        ]) + f_rot.apply(offset_vec)

        # 4. Combine rotations (Finger in Map * Wrist in Finger)
        off_quat = [finger_to_wrist_tf.transform.rotation.x, finger_to_wrist_tf.transform.rotation.y,
                    finger_to_wrist_tf.transform.rotation.z, finger_to_wrist_tf.transform.rotation.w]
        wrist_rot_map = f_rot * R.from_quat(off_quat)
        final_quat = wrist_rot_map.as_quat()

        # 5. Build the return message
        wrist_pose = PoseStamped()
        wrist_pose.header = finger_pose_map.header
        wrist_pose.pose.position.x = wrist_pos_map[0]
        wrist_pose.pose.position.y = wrist_pos_map[1]
        wrist_pose.pose.position.z = wrist_pos_map[2]
        wrist_pose.pose.orientation.x = final_quat[0]
        wrist_pose.pose.orientation.y = final_quat[1]
        wrist_pose.pose.orientation.z = final_quat[2]
        wrist_pose.pose.orientation.w = final_quat[3]

        return wrist_pose

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

        return self.get_traj(target_pose, side, traj_time)

    def get_traj(self, target_pose, side, traj_time):
        js = self.get_js()
        current_state = {jn: js[jn] for jn in group2joints[side]}
        q_target = self.compute_ik_with_retries(target_pose, current_state, side)
        if q_target is None:
            return None

        # We assume controller starts from current state, so single waypoint is fine
        traj = JointTrajectory()
        traj.joint_names = group2joints[side]

        current_q = self.get_js()

        point = JointTrajectoryPoint()
        point.positions = list(q_target)
        point.time_from_start.sec = int(traj_time)
        point.time_from_start.nanosec = int((traj_time % 1.0) * 1e9)

        traj.points.append(point)
        return traj

    def send_traj_and_get_handle(self, traj: JointTrajectory, side: str):
        self.controller_switcher.activate_controller(group2controller[side])

        print(f"##### TRAJECTORY\nfrom: {traj.points[0].positions}\nto:  {traj.points[-1].positions}\nnames: {traj.joint_names}\nlen(traj) = {len(traj.points)}")

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
                self.call_cli_sync(self.finger2srv["left"], RollerGripper.Request(finger_pos=2000))
                return True

            # optional small sleep to throttle loop
            self.ros_sleep(sleep_time)

        return False
    
    def plan_joint_goal(self, q, trajectory_time, side="left"):
        req = MotionPlanRequest()
        req.group_name = f"{side}_arm"  # Change to your move_group name
        req.num_planning_attempts = 1
        req.allowed_planning_time = 1.0

        req.pipeline_id = "chomp"

        current = self.current_q.copy()

        # Set Goal Constraints (Joint Space)
        constraints = Constraints()
        for name, val in zip(group2joints[side], q):
            req.start_state.joint_state.name.append(name)
            req.start_state.joint_state.position.append(current[name])

            jc = JointConstraint()
            jc.joint_name = name
            jc.position = val
            # jc.tolerance_above = 0.01
            # jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        req.goal_constraints.append(constraints)
 
        # Call service
        srv_req = GetMotionPlan.Request()
        srv_req.motion_plan_request = req
 
        future = self.plan_client.call_async(srv_req)
        rclpy.spin_until_future_complete(self, future)
 
        res = future.result()
        if res.motion_plan_response.error_code.val != MoveItErrorCodes.SUCCESS:
            print("PLANNING FAILED!")
            return None, None
        
        return res.motion_plan_response.trajectory_start, self.adjust_trajectory_timing(res.motion_plan_response.trajectory, trajectory_time)
 
    def publish_for_rviz(self, start_state, trajectory):
        display_msg = DisplayTrajectory()
        display_msg.model_id = "softenable_ur5e"

        display_msg.trajectory.append(trajectory)
        display_msg.trajectory_start = start_state

        self.traj_pub.publish(display_msg)
 
 
    def plan_cartesian_linear_path(self, waypoints, side="left"):
        """
        waypoints: A list of geometry_msgs/Pose objects
        """

        req = GetCartesianPath.Request()
        req.header.frame_id = "map" # Ensure this matches your robot base
        req.header.stamp = self.get_clock().now().to_msg()
        req.group_name = f"{side}_arm"
        req.link_name = f"{side}_arm_wrist_3_link"

       

        current = self.current_q.copy()
        for name in group2joints[side]:
            req.start_state.joint_state.name.append(name)
            req.start_state.joint_state.position.append(current[name])

        # The path will move linearly through these poses
        req.waypoints = waypoints
 
        # Max step: The distance (meters) between IK checks. 
        # Smaller = smoother but slower to compute.
        req.max_step = 0.01 
 
        # Jump threshold: Prevents joint "flips" (discontinuities in IK)
        req.jump_threshold = 0.0 
 
        req.avoid_collisions = True
 
        future = self.cartesian_path_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
 
        res = future.result()
 
        # Check 'fraction' to see if the full path was possible (1.0 = 100% complete)
        if res.fraction < 1.0:
            self.get_logger().warn(f"Only {res.fraction*100}% of the Cartesian path was planned.")
 
        return res

    def adjust_trajectory_timing(self, trajectory, target_duration):
        """
        Adjusts a ROS 2 RobotTrajectory to match a target_duration.
        """
        points = trajectory.joint_trajectory.points
        if not points:
            return trajectory

        # Get original duration in seconds
        # .nanosec / 1e9 handles the fractional part
        last_point_time = points[-1].time_from_start
        original_duration = last_point_time.sec + (last_point_time.nanosec / 1e9)

        if original_duration <= 0:
            return trajectory

        scaling_factor = target_duration / original_duration

        for point in points:
            # Calculate new time in seconds
            old_time_sec = point.time_from_start.sec + (point.time_from_start.nanosec / 1e9)
            new_time_sec = old_time_sec * scaling_factor
            
            # Convert float seconds -> rclpy.duration.Duration -> builtin_interfaces.msg.Duration
            point.time_from_start = Duration(seconds=new_time_sec).to_msg()

        return trajectory