import cv2
import time
import rclpy
import numpy as np

from PIL import Image
from cv_bridge import CvBridge

from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

from std_msgs.msg import Header, Int16MultiArray
from sensor_msgs.msg import CompressedImage

from stack_approach.helpers import publish_img
from stack_detect.helpers.dino_model import DINOModel, plot_boxes_to_image


class StackDetectorDINO(Node):

    def __init__(self, executor):
        self.exe = executor
        super().__init__("StackDetectorDINO")
        self.log = self.get_logger()

        self.bridge = CvBridge()
        self.cb_group = ReentrantCallbackGroup()

        self.declare_parameter('cpu_only', True)
        self.cpu_only = self.get_parameter("cpu_only").get_parameter_value().bool_value
   
        self.img_sub = self.create_subscription(
            CompressedImage, "/camera/color/image_raw/compressed", self.rgb_cb, 0, callback_group=MutuallyExclusiveCallbackGroup()
        )

        self.img_pub = self.create_publisher(CompressedImage, '/camera/color/dino/compressed', 0, callback_group=self.cb_group)
        self.boxpub = self.create_publisher(Int16MultiArray, '/stack_box', 10, callback_group=self.cb_group)

        ### DINO setup
        self.get_logger().info("setting up DINO ...")
        self.dino = DINOModel(self.cpu_only)
        self.get_logger().info("setup done!")

    def rgb_cb(self, msg): 
        ##### Convert
        img = self.bridge.compressed_imgmsg_to_cv2(msg)
        image_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), mode="RGB")
        
        ##### Run DINO
        dino_start = time.time()
        self.get_logger().info("running DINO ...")
        boxes_px, pred_phrases, confidences = self.dino.predict(
            image_pil, 
            "detect all stacks of clothing"
        )
        self.get_logger().info(f"DINO took {round(time.time()-dino_start,2)}s")

        image_with_box = plot_boxes_to_image(image_pil.copy(), boxes_px, pred_phrases)[0]

        ##### Publish
        self.boxpub.publish(Int16MultiArray(data=boxes_px[0]))
        publish_img(self.img_pub, image_with_box)


def main(args=None):
    """Creates subscriber node and spins it"""
    rclpy.init(args=args)

    executor = MultiThreadedExecutor(num_threads=4)
    node = StackDetectorDINO(executor=executor)

    executor.add_node(node)

    try:
        node.get_logger().info('Starting node, shut down with CTRL-C')
        rclpy.spin(node, executor)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down.\n')

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
