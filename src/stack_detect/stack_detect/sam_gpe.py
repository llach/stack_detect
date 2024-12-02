import cv2
import time
import rclpy
import numpy as np

from PIL import Image
from threading import Lock
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from std_msgs.msg import Header
from sensor_msgs.msg import Image as ImageMSG, CompressedImage, CameraInfo

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import PointStamped

from stack_approach.helpers import publish_img, grasp_pose_to_wrist
from stack_detect.helpers.sam2_model import SAM2Model
from stack_detect.helpers.dino_model import DINOModel
from stack_msgs.srv import MoveArm, GripperService
from stack_approach.grasping_primitives import direct_approach_grasp

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

    def info_cb(self, msg): self.K = np.array(msg.k).reshape(3, 3)
            
    def depth_cb(self, msg):
        with self.depth_lock:
            self.depth_msg = msg

    def rgb_cb(self, msg): 
        with self.img_lock:
            self.img_msg = msg

    def extract_grasp_point(self): 
        while self.img_msg is None or self.depth_msg is None or self.K is None: 
            time.sleep(0.05)
            rclpy.spin_once(self)

        ##### Convert image
        with self.img_lock:
            with self.depth_lock:
                depth_img = self.bridge.imgmsg_to_cv2(self.depth_msg)
                img = self.bridge.compressed_imgmsg_to_cv2(self.img_msg)
        img_raw = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(img_raw, mode="RGB")

        ##### Run DINO
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

        should_save = input("grasp? [Y/n]")

        if should_save.strip().lower() == "n":
            print("not grasping. bye.")
            return
        
        grasp_pose_wrist = grasp_pose_to_wrist(self.tf_buffer, center_point)
        direct_approach_grasp(self, self.move_cli, self.gripper_cli, grasp_pose_wrist)

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
