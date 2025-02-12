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

from std_msgs.msg import Header
from sensor_msgs.msg import Image as ImageMSG, CompressedImage, CameraInfo

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import PointStamped, PoseStamped
from datetime import datetime
from stack_msgs.action import RecordCloud
from stack_msgs.srv import StoreData
from stack_approach.helpers import grasp_pose_to_wrist, publish_img, point_to_pose, empty_pose, get_trafo, inv, matrix_to_pose_msg, pose_to_matrix, call_cli_sync
from stack_detect.helpers.sam2_model import SAM2Model
from stack_detect.helpers.dino_model import DINOModel
from stack_msgs.srv import MoveArm, GripperService
from stack_approach.grasping_primitives import direct_approach_grasp, angled_approach_grasp
from scipy.spatial.transform import Rotation as R

class SAMGraspPointExtractor(Node):

    def __init__(self):
        super().__init__("SAMGraspPointExtractor")
        self.log = self.get_logger()

        self.bridge = CvBridge()
        self.cbg = ReentrantCallbackGroup()

        self.declare_parameter('dino_cpu', True)
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
        ##### wait for data and tfs
        self.get_logger().info("waiting for data ...")
        while self.img_msg is None or self.depth_msg is None or self.K is None: 
            time.sleep(0.05)
            rclpy.spin_once(self)

        self.get_logger().info("waiting for transforms ...")
        while not (
            self.tf_buffer.can_transform("map", "wrist_3_link", rclpy.time.Time()) and
            self.tf_buffer.can_transform("map", "camera_color_optical_frame", rclpy.time.Time())
        ):
            time.sleep(0.05)
            rclpy.spin_once(self)
        
        self.get_logger().info("ready to grasp!")
        
        ##### Convert image
        with self.img_lock:
            with self.depth_lock:
                depth_img = self.bridge.imgmsg_to_cv2(self.depth_msg)
                img = self.bridge.compressed_imgmsg_to_cv2(self.img_msg)
        img_raw = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(img_raw, mode="RGB")

        #### Run DINO
        dino_start = time.time()
        self.get_logger().info("running DINO ...")
        boxes_px, pred_phrases, confidences = self.dino.predict(
            image_pil, 
            "detect all stacks of clothing"
        )
        self.get_logger().info(f"DINO took {round(time.time()-dino_start,2)}s")

        ##### Run SAM
        sam_start = time.time()
        self.get_logger().info("running SAM ...")
        masks = self.sam.predict(img_raw)
        self.get_logger().info(f"SAM took {round(time.time()-sam_start,2)}s")

        img_overlay, _, line_center = SAM2Model.detect_stack(img, masks, boxes_px[0])
        publish_img(self.img_pub, cv2.cvtColor(img_overlay, cv2.COLOR_RGB2BGR))

        if line_center is None:
            self.get_logger().warn("Layer not found!")
            return

        # get 3D point and publish
        center_point = SAM2Model.get_center_point(line_center, depth_img, self.K)
        self.grasp_point_pub.publish(center_point)
        

        should_grasp = input("grasp? [d/a/Q]")
        sg = should_grasp.strip().lower()
        
        if sg not in ["a", "d"]:
            print("not grasping. bye.")
            return
        
        gh = self.start_recording()
        time.sleep(0.3)
        
        # if sg == "d":
        #     grasp_pose_wrist = grasp_pose_to_wrist(self.tf_buffer, center_point, x_off=0.017)
        #     self.grasp_pose_pub.publish(grasp_pose_wrist)

        #     direct_approach_grasp(self, self.move_cli, self.gripper_cli, grasp_pose_wrist, with_grasp=False)
        # elif sg == "a":
        
        #     Tfw = get_trafo("finger", "wrist_3_link", self.tf_buffer)
            
        #     center_pose = pose_to_matrix(self.tf_buffer.transform(point_to_pose(center_point), "wrist_3_link"))
        #     center_pose[:3,:3] = np.eye(3) @ R.from_euler("xyz", [0, -30, 0], degrees=True).as_matrix() 
        #     center_pose = center_pose @ Tfw
        #     # center_pose[:3,3] += [0.0099,0,-0.015] # dark stack
        #     center_pose[:3,3] += [0.0065,0,-0.015] # light stack
        #     print(center_pose)
    
            
        #     center_pose_msg = matrix_to_pose_msg(center_pose, "wrist_3_link")
        #     self.grasp_pose_pub.publish(center_pose_msg)
            
        #     angled_approach_grasp(self, self.move_cli, self.gripper_cli, center_pose_msg, self.tf_buffer, with_grasp=True)            
        

        
        
        #### Store data
        time.sleep(0.3)
        self.stop_recording(gh)
        should_save = input("save? [Y/n]")

        if should_save.strip().lower() == "n":
            print("not saving data. bye.")
            return

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

    try:
        node.extract_grasp_point()
    except KeyboardInterrupt:
        print("Keyboard Interrupt!")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
