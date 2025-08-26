import os
import cv2
import time
import rclpy
import pickle
import numpy as np

from PIL import Image
from threading import Lock, Event
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from builtin_interfaces.msg import Time
from std_msgs.msg import Header
from sensor_msgs.msg import Image as ImageMSG, CompressedImage, CameraInfo

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import PointStamped, PoseStamped
from datetime import datetime
from stack_msgs.action import RecordCloud
from stack_msgs.srv import StoreData, DINOSrv
from stack_approach.helpers import grasp_pose_to_wrist, publish_img, point_to_pose, empty_pose, get_trafo, inv, matrix_to_pose_msg, pose_to_matrix, call_cli_sync
from stack_detect.helpers.sam2_model import SAM2Model
from stack_detect.helpers.dino_model import DINOModel, plot_boxes_to_image
from stack_msgs.srv import StackDetect
# from stack_approach.grasping_primitives import direct_approach_grasp, angled_approach_grasp, dark_stack_roller, thin_stack_roller
from scipy.spatial.transform import Rotation as R

class SAMGraspPointExtractor(Node):

    def __init__(self):
        super().__init__("SAMGraspPointExtractor")
        self.log = self.get_logger()

        self.bridge = CvBridge()
        self.cbg = ReentrantCallbackGroup()

        self.img_ready = Event()
        self.depth_ready = Event()
        self.info_ready = Event()

        self.declare_parameter('dino_cpu', False)
        self.dino_cpu = self.get_parameter("dino_cpu").get_parameter_value().bool_value

        self.sam = SAM2Model(checkpoint = f"/home/ros/pretrained/sam2/sam2.1_hiera_large.pt", model_cfg = "//home/ros/pretrained/sam2/sam2.1_hiera_l.yaml")
        self.dino = DINOModel(prefix="/home/ros/pretrained/dino/",cpu_only=self.dino_cpu)

        self.img_lock = Lock()
        self.depth_lock = Lock()
        self.img_msg, self.depth_msg, self.K = None, None, None

        self.depth_sub = self.create_subscription(
            ImageMSG, "/camera/aligned_depth_to_color/image_raw", self.depth_cb, 0, callback_group=self.cbg
        )
        self.img_sub = self.create_subscription(
            CompressedImage, "/camera/color/image_raw/compressed", self.rgb_cb, 0, callback_group=self.cbg
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/color/camera_info', self.info_cb, 0
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.img_pub = self.create_publisher(CompressedImage, '/camera/color/sam/compressed', 0, callback_group=self.cbg)
        self.grasp_point_pub = self.create_publisher(PointStamped, '/grasp_point', 10, callback_group=self.cbg)
        self.grasp_pose_pub = self.create_publisher(PoseStamped, '/debug_grasp_pose', 10, callback_group=self.cbg)

        self.srv = self.create_service(StackDetect, '/stack_detect', self.extract_grasp_point)

        print("stack detect service running!")

    def depth_cb(self, msg):
        with self.depth_lock:
            self.depth_msg = msg
            self.depth_ready.set()

    def rgb_cb(self, msg): 
        with self.img_lock:
            self.img_msg = msg
            self.img_ready.set()

    def info_cb(self, msg): 
        self.K = np.array(msg.k).reshape(3, 3)
        self.info_ready.set()

    def wait_for_data(self, timeout=5.0):
        self.get_logger().info("Waiting for image, depth, and camera info...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.img_ready.is_set() and self.depth_ready.is_set() and self.info_ready.is_set():
                self.get_logger().info("All data received.")
                return True
            time.sleep(0.05)  # passive wait, lets executor run
        self.get_logger().warn("Timeout while waiting for data.")
        return False
            
    def start_recording(self):
        # Create a goal to start data collection
        goal_msg = RecordCloud.Goal()
        goal_msg.start = True  # Set the start flag

        # Send the goal
        self.get_logger().info('Sending goal to start data collection...')
        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        rclpy.spin_until_future_complete(self, send_goal_future)

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().fatal('Goal rejected by action server.')
            exit(0)

        self.get_logger().info('Goal accepted by action server.')
        return goal_handle

    def feedback_callback(self, feedback_msg):
        self.n_samples = np.array(feedback_msg.feedback.n_samples)

    def stop_recording(self, gh):
        cancel_future = gh.cancel_goal_async()
        rclpy.spin_until_future_complete(self, cancel_future)

        cancel_response = cancel_future.result()
        if cancel_response:
            self.get_logger().info('Action successfully canceled.')
        else:
            self.get_logger().info('Failed to cancel the action.')
            exit(0)

    def extract_grasp_point(self, req, res): 
        self.wait_for_data()
        
        ##### Convert image
        with self.img_lock:
            with self.depth_lock:
                depth_img = self.bridge.imgmsg_to_cv2(self.depth_msg)
                img = self.bridge.compressed_imgmsg_to_cv2(self.img_msg)
        img_raw = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(img_raw, mode="RGB")

        publish_img(self.img_pub, np.array(img_raw))

        #### Run DINO
        dino_start = time.time()
        self.get_logger().info("running DINO ...")
        boxes_px, pred_phrases, confidences = self.dino.predict(
            image_pil, 
            "detect all stacks of clothing"
        )
        self.get_logger().info(f"DINO took {round(time.time()-dino_start,2)}s")
        
        image_with_box = plot_boxes_to_image(image_pil.copy(), boxes_px, pred_phrases)[0]
        publish_img(self.img_pub, np.array(image_with_box))
        
        # box_idx = input("which box? [0]")
        box_idx = 0
        try:
            if box_idx == "": box_idx = "0"
            box_idx = int(box_idx)
            box = boxes_px[box_idx]
        except Exception as e:
            print(f"box indexing error: {e}")
            res.success = False
            return res

        ##### Run SAM
        sam_start = time.time()
        self.get_logger().info("running SAM ...")
        masks = self.sam.predict(img_raw)
        self.get_logger().info(f"SAM took {round(time.time()-sam_start,2)}s")

        img_overlay, _, line_center = SAM2Model.detect_stack(img, masks, box)
        publish_img(self.img_pub, img_overlay)

        if line_center is None:
            self.get_logger().warn("Layer not found!")
            res.success = False
            return res

        # get 3D point and publish
        center_point = SAM2Model.get_center_point(line_center, depth_img, self.K) # center point is stamped in camera coordinates

        grasp_pose_wrist = grasp_pose_to_wrist(self.tf_buffer, center_point, x_off=0.045, z_off=-0.20) # FRANKENROLLER

        GOAL_ANGLE = 50 # in degrees
        OFFSET = [0.007,0,-0.015]
        MAP_OFFSET = [0,-0.035,-0.005]

        self.wait_for_data()

        Tfw = get_trafo("right_finger", "right_arm_wrist_3_link", self.tf_buffer)

        grasp_pose_wrist = grasp_pose_to_wrist(self.tf_buffer, center_point, x_off=0, z_off=0)
        grasp_pose_finger = self.tf_buffer.transform(grasp_pose_wrist, "right_finger")                # grasp pose in finger frame
        grasp_pose_finger_mat = pose_to_matrix(grasp_pose_finger)

        Tmf = get_trafo("map", "right_finger", self.tf_buffer)
        Tmf[:3,3] = Tmf[:3,3] + MAP_OFFSET
        current_angle_offset = np.rad2deg(np.arccos(np.dot([0,1,0], Tmf[:3,:3]@[0,0,1]))) # angle between ground plane and z axis in wrist / finger frame
        print(current_angle_offset)

        grasp_pose_finger_mat[:3,:3] = grasp_pose_finger_mat[:3,:3] @ R.from_euler("xyz", [0, -(GOAL_ANGLE - current_angle_offset), 0], degrees=True).as_matrix()
        # Copy the rotated pose
        grasp_pose_wrist_mat = grasp_pose_finger_mat.copy()

        # Apply translation in the rotated/local frame
        translation_local = grasp_pose_finger_mat[:3,:3] @ (Tfw[:3,3] + OFFSET)
        grasp_pose_wrist_mat[:3,3] += translation_local

        goal_finger = matrix_to_pose_msg(grasp_pose_wrist_mat, "right_finger")

        goal_map = self.tf_buffer.transform(goal_finger, "map")
        goal_map.pose.position.x += MAP_OFFSET[0]
        goal_map.pose.position.y += MAP_OFFSET[1]
        goal_map.pose.position.z += MAP_OFFSET[2]

        goal_wrist = self.tf_buffer.transform(goal_map, "right_arm_wrist_3_link")

        # goal_wrist.pose.position.x += 0.025
        goal_wrist.pose.position.z += 0.01

        res.success = True
        res.target_pose = goal_wrist
        return res


def main(args=None):
    """Creates subscriber node and spins it"""
    rclpy.init(args=args)

    node = SAMGraspPointExtractor()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        print("Keyboard Interrupt!")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
