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
from sensor_msgs.msg import Image as ImageMSG, CompressedImage

from stack_approach.helpers import publish_img
from stack_detect.helpers.sam2_model import SAM2Model
from stack_detect.helpers.dino_model import DINOModel


class SAMGraspPointExtractor(Node):

    def __init__(self):
        super().__init__("SAMGraspPointExtractor")
        self.log = self.get_logger()

        self.bridge = CvBridge()

        self.declare_parameter('dino_cpu', True)
        self.dino_cpu = self.get_parameter("dino_cpu").get_parameter_value().bool_value

        self.sam = SAM2Model()
        self.dino = DINOModel(cpu_only=self.dino_cpu)

        self.rgb_msg = None
        self.rgb_lck = Lock()

        self.cbg = ReentrantCallbackGroup()
        self.img_sub = self.create_subscription(
            CompressedImage, "/camera/color/image_raw/compressed", self.rgb_cb, 0, callback_group=self.cbg
        )

        self.img_pub = self.create_publisher(CompressedImage, '/camera/color/sam/compressed', 0, callback_group=self.cbg)

    def rgb_cb(self, msg):
        with self.rgb_lck:
            self.rgb_msg = msg 

    def extract_grasp_point(self): 
        while self.rgb_msg is None: 
            time.sleep(0.05)
            rclpy.spin_once(self)

        ##### Convert image
        with self.rgb_lck:
            img_raw = cv2.cvtColor(self.bridge.compressed_imgmsg_to_cv2(self.rgb_msg), cv2.COLOR_BGR2RGB)
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

        img_overlay, line_pixels, line_center = SAM2Model.detect_stack(img_raw, masks, boxes_px[0])

        publish_img(self.img_pub, img_overlay)



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
