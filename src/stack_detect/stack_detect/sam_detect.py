import cv2
import time
import rclpy
import numpy as np

from cv_bridge import CvBridge
from threading import Lock

from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

from std_msgs.msg import Int16MultiArray
from sensor_msgs.msg import CompressedImage, CameraInfo, Image as ImageMSG
from geometry_msgs.msg import PointStamped

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import PointStamped

from stack_approach.helpers import publish_img, pixel_to_point
from stack_detect.helpers.sam2_model import SAM2Model

class StackDetectorSAM(Node):

    def __init__(self, executor):
        self.exe = executor
        super().__init__("StackDetectorSAM")
        self.log = self.get_logger()

        self.bridge = CvBridge()
        self.cb_group = ReentrantCallbackGroup()

        self.depth_sub = self.create_subscription(
            ImageMSG, "/camera/aligned_depth_to_color/image_raw", self.depth_cb, 0, callback_group=MutuallyExclusiveCallbackGroup()
        )
        self.img_sub = self.create_subscription(
            CompressedImage, "/camera/color/image_raw/compressed", self.rgb_cb, 0, callback_group=MutuallyExclusiveCallbackGroup()
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/color/camera_info', self.info_cb, 0
        )
        self.box_sub = self.create_subscription(
            Int16MultiArray, '/stack_box', self.box_cb, 0, callback_group=ReentrantCallbackGroup()    
        )

        self.img_pub = self.create_publisher(CompressedImage, '/camera/color/sam/compressed', 0, callback_group=self.cb_group)
        self.ppub = self.create_publisher(PointStamped, '/grasp_point', 10, callback_group=self.cb_group)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.depth_lock = Lock()
        self.depth_msg = None
        self.img_lock = Lock()
        self.img_msg = None
        self.processing = False

        ### SAM setup
        self.get_logger().info("loading model")
        self.sam = SAM2Model()
        self.get_logger().info("setup done!")

    def info_cb(self, msg): self.K = np.array(msg.k).reshape(3, 3)
            
    def depth_cb(self, msg):
        with self.depth_lock:
            self.depth_msg = msg

    def rgb_cb(self, msg): 
        with self.img_lock:
            self.img_msg = msg

    def box_cb(self, msg): 
        self.log.info("got new box!")
        
        if self.depth_msg is None:
            self.get_logger().warn("no depth image yet ...")
            return
        
        if self.img_msg is None:
            self.get_logger().warn("no rgb image yet ...")
            return
        
        if self.processing:
            print("still working ...")
            return
        self.processing = True
        
        ##### Convert RGB image
        with self.img_lock:
            with self.depth_lock:
                depth_img = self.bridge.imgmsg_to_cv2(self.depth_msg)
                img = self.bridge.compressed_imgmsg_to_cv2(self.img_msg)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        ##### Run SAM
        sam_start = time.time()
        self.get_logger().info("running SAM ...")
        masks = self.sam.predict(img)
        self.get_logger().info(f"SAM took {round(time.time()-sam_start,2)}s")

        img_overlay, line_pixels, line_center = SAM2Model.detect_stack(img, masks, msg.data)
        line_dist = depth_img[*line_center]/1000 # TODO maybe indices are the other way round. also factor 1000 correct?

        center_point = pixel_to_point(line_center, line_dist, self.K)
        center_point = self.tf_buffer.transform(center_point, "map", timeout=rclpy.duration.Duration(seconds=5))

        ##### Publish
        publish_img(self.img_pub, cv2.cvtColor(img_overlay, cv2.COLOR_RGB2BGR))
        self.ppub.publish(center_point)

        self.processing = False
        return

def main(args=None):
    """Creates subscriber node and spins it"""
    rclpy.init(args=args)

    executor = MultiThreadedExecutor(num_threads=4)
    node = StackDetectorSAM(executor=executor)

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
