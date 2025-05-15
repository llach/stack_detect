import os
import cv2
import time
import rclpy
import pickle
import numpy as np

from PIL import Image
from threading import Lock
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
from stack_msgs.srv import MoveArm, GripperService
from stack_approach.grasping_primitives import direct_approach_grasp, angled_approach_grasp
from scipy.spatial.transform import Rotation as R

class SAMGraspPointExtractor(Node):

    def __init__(self):
        super().__init__("SAMGraspPointExtractor")
        self.log = self.get_logger()

        self.bridge = CvBridge()
        self.cbg = ReentrantCallbackGroup()

        self.declare_parameter('dino_cpu', False)
        self.dino_cpu = self.get_parameter("dino_cpu").get_parameter_value().bool_value

        self.sam = SAM2Model()
        self.dino = DINOModel(cpu_only=self.dino_cpu)

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

        self.move_cli = self.create_client(MoveArm, "move_arm")
        while not self.move_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('move arm service not available, waiting again...')

        self.gripper_cli = self.create_client(GripperService, "gripper")
        while not self.gripper_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('gripper service not available, waiting again...')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.img_pub = self.create_publisher(CompressedImage, '/camera/color/sam/compressed', 0, callback_group=self.cbg)
        self.grasp_point_pub = self.create_publisher(PointStamped, '/grasp_point', 10, callback_group=self.cbg)
        self.grasp_pose_pub = self.create_publisher(PoseStamped, '/debug_grasp_pose', 10, callback_group=self.cbg)
        
        self._action_client = ActionClient(self, RecordCloud, 'collect_cloud_data', callback_group=ReentrantCallbackGroup())
        while not self._action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('data collection action not available, waiting again...')
            
        self.store_cli = self.create_client(StoreData, "store_cloud_data")
        while not self.store_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('store data service not available, waiting again...')

    def info_cb(self, msg): self.K = np.array(msg.k).reshape(3, 3)
            
    def depth_cb(self, msg):
        with self.depth_lock:
            self.depth_msg = msg

    def rgb_cb(self, msg): 
        with self.img_lock:
            self.img_msg = msg

    def wait_for_data(self):
        ##### wait for data and tfs
        self.get_logger().info("waiting for data ...")
        while self.img_msg is None or self.depth_msg is None or self.K is None: 
            time.sleep(0.05)
            rclpy.spin_once(self)

        self.get_logger().info("waiting for transforms ...")
        while not (
            self.tf_buffer.can_transform("map", "wrist_3_link", rclpy.time.Time()) and
            self.tf_buffer.can_transform("wrist_3_link", "map", rclpy.time.Time()) and
            self.tf_buffer.can_transform("map", "camera_color_optical_frame", rclpy.time.Time())
        ):
            time.sleep(0.05)
            rclpy.spin_once(self)
        
        self.get_logger().info("ready to grasp!")
            
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

    def extract_grasp_point(self): 
        with self.img_lock:
            with self.depth_lock:
                self.depth_msg = None
                self.img_msg = None

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
        
        box_idx = input("which box? ")
        try:
            if box_idx == "": box_idx = "0"
            box_idx = int(box_idx)
            box = boxes_px[box_idx]
        except Exception as e:
            print(f"box indexing error: {e}")
            return

        ##### Run SAM
        sam_start = time.time()
        self.get_logger().info("running SAM ...")
        masks = self.sam.predict(img_raw)
        self.get_logger().info(f"SAM took {round(time.time()-sam_start,2)}s")

        img_overlay, _, line_center = SAM2Model.detect_stack(img, masks, box)
        publish_img(self.img_pub, img_overlay)

        if line_center is None:
            self.get_logger().warn("Layer not found!")
            return

        # get 3D point and publish
        center_point = SAM2Model.get_center_point(line_center, depth_img, self.K) # center point is stamped in camera coordinates
        self.grasp_point_pub.publish(center_point)
        
        should_grasp = input("grasp? [d/a/Q]")
        sg = should_grasp.strip().lower()
        
        if sg not in ["a", "d", ""]:
            print("not grasping. bye.")
            return
        
        gh = self.start_recording()
        time.sleep(0.3)

        # transform grasp point to pose stamped in wrist3link, apply offsets in wrist space
        # grasp_pose_wrist = grasp_pose_to_wrist(self.tf_buffer, center_point, x_off=0.018, z_off=-0.22) # HAND-E
        # grasp_pose_wrist = grasp_pose_to_wrist(self.tf_buffer, center_point, x_off=0.024, z_off=-0.262) # BLUE GRIPPER
        grasp_pose_wrist = grasp_pose_to_wrist(self.tf_buffer, center_point, x_off=0.035, z_off=-0.2) # FRANKENROLLER

        if sg == "d" or sg == "":
            self.grasp_pose_pub.publish(grasp_pose_wrist)

            direct_approach_grasp(self, self.move_cli, self.gripper_cli, grasp_pose_wrist, with_grasp=False)
        elif sg == "a":
            GOAL_ANGLE = 45 # in degrees
            OFFSET = [0.05,0,-0.1]

            self.wait_for_data()

            Tfw = get_trafo("finger", "wrist_3_link", self.tf_buffer)

            grasp_pose_wrist = grasp_pose_to_wrist(self.tf_buffer, center_point, x_off=0, z_off=0)
            grasp_pose_finger = self.tf_buffer.transform(grasp_pose_wrist, "finger")                # grasp pose in finger frame
            grasp_pose_finger_mat = pose_to_matrix(grasp_pose_finger)

            Tmf = get_trafo("map", "finger", self.tf_buffer)
            current_angle_offset = np.rad2deg(np.arccos(np.dot([0,1,0], Tmf[:3,:3]@[0,0,1]))) # angle between ground plane and z axis in wrist / finger frame
            print(current_angle_offset)

            grasp_pose_finger_mat[:3,:3] = grasp_pose_finger_mat[:3,:3] @ R.from_euler("xyz", [0, -(GOAL_ANGLE - current_angle_offset), 0], degrees=True).as_matrix()
            # Copy the rotated pose
            grasp_pose_wrist_mat = grasp_pose_finger_mat.copy()

            # Apply translation in the rotated/local frame
            translation_local = grasp_pose_finger_mat[:3,:3] @ (Tfw[:3,3] + OFFSET)
            grasp_pose_wrist_mat[:3,3] += translation_local


            goal_finger = matrix_to_pose_msg(grasp_pose_wrist_mat, "finger")
            goal_wrist = self.tf_buffer.transform(goal_finger, "wrist_3_link")         

            print("publishing goal ...")
            self.grasp_pose_pub.publish(goal_wrist)

            # # 1st: how much above the stack? positive = up $##### OLD CODE
            # # 3rd: how much into the stack? negative = further away from stack
            # # center_pose[:3,3] += [0.0099,0,-0.015] # dark stack
            # # center_pose[:3,3] += [0.00,0,-0.03] # weird light stack BEST FOR BLUE GRIPPER
            # # center_pose[:3,3] += [0.0065,0,-0.025] # light stack
            
            # angled_approach_grasp(self, self.move_cli, self.gripper_cli, wrist_post_wrist, self.tf_buffer, with_grasp=False)
        
        #### Store data
        time.sleep(0.3)
        self.stop_recording(gh)
        time.sleep(0.5)
        # should_save = input("save? [Y/n]")

        # if should_save.strip().lower() == "n":
        #     print("not saving data. bye.")
        #     return

        if False:
            print("saving data ...")
            
            sample_dir = f"{os.environ['HOME']}/repos/unstack_deliverable/" + datetime.now().strftime("%Y.%m.%d_%H.%M.%S") + "/"
            os.makedirs(sample_dir)
            
            image_pil.save(f"{sample_dir}/raw_image.png")
            Image.fromarray(img_overlay, mode="RGB").save(f"{sample_dir}/sam_output.png")

            print("store request")
            store_req = StoreData.Request()
            store_req.dir = sample_dir
            call_cli_sync(self, self.store_cli, store_req)

        self.get_logger().info("all done!")


def main(args=None):
    """Creates subscriber node and spins it"""
    rclpy.init(args=args)

    node = SAMGraspPointExtractor()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    node.wait_for_data()
    try:
        while True:
            node.extract_grasp_point()
            again = input("again? [Y/n]").strip().lower()
            if again == "y" or again == "":
                continue
            else: break
    except KeyboardInterrupt:
        print("Keyboard Interrupt!")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
