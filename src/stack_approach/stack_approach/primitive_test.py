#!/usr/bin/env python3
import os
import time
import rclpy
import numpy as np
from datetime import datetime

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
from cv_bridge import CvBridge
from stack_approach.helpers import transform_to_pose_stamped, publish_img, point_to_pose, empty_pose, get_trafo, inv, matrix_to_pose_msg, pose_to_matrix, call_cli_sync
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CompressedImage
from threading import Lock
from PIL import Image


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
        self.bridge = CvBridge()
        self.controller_switcher = ControllerSwitcher()

        self.img_lock = Lock()
        self.other_img_lock = Lock()
        self.img_msg, self.other_img_msg = None, None

        self.img_sub = self.create_subscription(
            CompressedImage, "/camera/color/image_raw/compressed", self.rgb_cb, 0, callback_group=self.recbg
        )

        self.img_sub = self.create_subscription(
            CompressedImage, "/unfolding_camera/color/image_raw/compressed", self.other_rgb_cb, 0, callback_group=self.recbg
        )
        
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

        self.sam_client = self.create_client(StackDetect, "stack_detect")
        while not self.sam_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('SAM service not available, waiting again...')

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

    def rgb_cb(self, msg): 
        with self.img_lock:
            self.img_msg = msg

    def other_rgb_cb(self, msg): 
        with self.other_img_lock:
            self.other_img_msg = msg

    def wait_for_data(self, timeout=5.0):
        self.get_logger().info("Waiting for data ...")

        self.img_msg, self.other_img_msg = None, None

        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.img_msg and self.other_img_msg:
                self.get_logger().info("All data received.")
                return True
            time.sleep(0.05)  # passive wait, lets executor run
        self.get_logger().warn("Timeout while waiting for data.")
        return False

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

    def go_to_q(self, q, time=3):
        fut = self.execute_traj(
            "right", 
            np.array([float(time)]), 
            np.array([q])
        )
        await_action_future(self, fut)

    def go_to_pose(self, pose, time):
        fut = self.move_cli.call_async(MoveArm.Request(
            target_pose = pose,
            execution_time = float(time),
            execute = True,
            controller_name = "right_arm_joint_trajectory_controller",
            ik_link = "right_arm_wrist_3_link",
            name_target = ["right_arm_shoulder_pan_joint", "right_arm_shoulder_lift_joint", "right_arm_elbow_joint", "right_arm_wrist_1_joint", "right_arm_wrist_2_joint", "right_arm_wrist_3_joint"]
        ))
        rclpy.spin_until_future_complete(self, fut)


    def move_rel(self, x=0.0, y=0.0, z=0.0, time=5):
        msg = transform_to_pose_stamped(self.tf_buffer.lookup_transform("map", "right_arm_wrist_3_link", rclpy.time.Time()))

        msg.pose.position.x += x
        msg.pose.position.y += y
        msg.pose.position.z += z

        self.go_to_pose(msg, time)

    def ros_sleep(self, sec):
        for _ in range(int(sec/0.1)):
            time.sleep(0.1)
            rclpy.spin_once(self)

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
    node.call_cli_sync(node.finger2srv["left"], RollerGripper.Request(roller_vel=80, roller_duration=5.0))
    node.call_cli_sync(node.finger2srv["left"], RollerGripper.Request(finger_pos=3500))

def main(args=None):
    run_name = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    store_base = f"/home/ros/ws/src/bag_opening/gg_full_trials/{run_name}_unstacking"

    # GOAL_ANGLE = 55 # THIN degrees
    # TRANS_OFFSET_MAP = [0, -0.05, 0.011] # THIN meters

    GOAL_ANGLE = 45 # NORMAL degrees
    TRANS_OFFSET_MAP = [0, -0.05, 0.021] # NORMAL meters

    rclpy.init(args=args)
    node = TrajectoryPublisher()

    node.wait_for_data()
    
    node.call_cli_sync(node.finger2srv["right"], RollerGripper.Request(finger_pos=2500))
    node.start_pose(time=3)
    node.ros_sleep(1)

    while True:
        with node.img_lock:
            img_b4 = node.bridge.compressed_imgmsg_to_cv2(node.img_msg, "rgb8")

        fut = node.sam_client.call_async(StackDetect.Request())
        rclpy.spin_until_future_complete(node, fut)

        stack_pose = fut.result().target_pose

        stack_mat_in_finger = pose_to_matrix(node.tf_buffer.transform(stack_pose, "right_finger"))
        stack_mat_in_finger[:3,:3] = np.eye(3)

        Tmf = get_trafo("map", "right_finger", node.tf_buffer)
        Tfw = get_trafo("right_finger", "right_arm_wrist_3_link", node.tf_buffer)

        current_angle_offset = np.rad2deg(np.arccos(np.dot([0,1,0], Tmf[:3,:3]@[0,0,1]))) # angle between ground plane and z axis in wrist / finger frame
        print(f"map Y <-> finger Z {current_angle_offset:.2f}deg")

        goal_mat_finger = stack_mat_in_finger.copy()
        goal_mat_finger[:3,:3] = goal_mat_finger[:3,:3] @ R.from_euler("xyz", [0, -(GOAL_ANGLE - current_angle_offset), 0], degrees=True).as_matrix()
        
        goal_mat_finger_in_map = Tmf @ goal_mat_finger
        goal_mat_finger_in_map[:3,3] += TRANS_OFFSET_MAP

        goal_mat_wrist_in_map = goal_mat_finger_in_map @ Tfw

        goal_pose_finger_in_map = matrix_to_pose_msg(goal_mat_finger_in_map, "map")
        goal_pose_wrist_in_map  = matrix_to_pose_msg(goal_mat_wrist_in_map, "map")

        node.finger_pose_pub.publish(goal_pose_finger_in_map)
        node.wrist_pose_pub.publish(goal_pose_wrist_in_map)

        inp = input("####\ngood?").lower().strip()
        if inp == "q":
            return
        elif inp == "y":
            break
    
    print(f"\n\n{run_name}\n\n")

    node.go_to_pose(goal_pose_wrist_in_map, 3)

    # return
    node.move_rel(y=0.03, z=-0.01, time=.6)
    node.move_rel(y=0.035, time=.6)

    # node.move_rel(y=0.055, time=1)

    node.move_rel(z=0.007, time=.3)
    node.move_rel(y=0.025, time=.3)

    node.call_cli_sync(node.finger2srv["right"], RollerGripper.Request(finger_pos=800))
    node.ros_sleep(0.2)

    return

    with node.other_img_lock:
        img_grasp = node.bridge.compressed_imgmsg_to_cv2(node.other_img_msg, "rgb8") if node.other_img_msg else None

    input("conf?")
    
    node.start_pose(time=2)

    node.go_to_q([-1.1826,-0.669801,1.29451,-1.57216,0.941826,4.03527], time=2)
    node.go_to_q([-0.829086,-1.01202,1.98262,-2.05671,0.951,3.88346], time=1.5)
    node.go_to_q([-0.731177,-1.07293,1.9299,-1.88504,0.999185,3.78101], time=0.5)

    node.call_cli_sync(node.finger2srv["right"], RollerGripper.Request(finger_pos=2500))
    
    node.go_to_q([-0.679236,-1.10291,1.97395,-1.87112,1.0267,3.72891], time=0.25)
    node.go_to_q([-0.740686,-1.06235,2.12228,-2.09265,0.994247,3.7895], time=0.4)


    node.go_to_q([-1.10223,-0.746435,2.00881,-2.5518,0.8478,4.20552], time=1)

    node.go_to_q([-1.4472,-0.5199,2.0481,-2.9698,0.7930,-0.0089], time=2)

    node.ros_sleep(1.5)

    with node.img_lock:
        img_table = node.bridge.compressed_imgmsg_to_cv2(node.img_msg, "rgb8")

    os.makedirs(store_base, exist_ok=True)
    Image.fromarray(img_b4).save(f"{store_base}/img_before.png")
    Image.fromarray(img_grasp).save(f"{store_base}/img_grasp.png")
    Image.fromarray(img_table).save(f"{store_base}/img_table.png")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
