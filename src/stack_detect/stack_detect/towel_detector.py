"""Subscriber module"""
import cv2
import rclpy
import numpy as np

from rclpy.node import Node
from cv_bridge import CvBridge 

from sensor_msgs.msg import Image

from stack_detect.detectors import ColorLineDetector, LINE_DEBUG_TYPE


class TowelDetector(Node):
    """Subscriber node"""

    def __init__(self):
        super().__init__("towel_detector")

        self.declare_parameter("debug_img", False)
        self.debug_img = self.get_parameter("debug_img").get_parameter_value().bool_value

        self.subscription = self.create_subscription(
            Image, "/video_in", self.listener_callback, 0
        )

        if self.debug_img:
            self.img_pub = self.create_publisher(Image, "debug_img", 10)

        self.br = CvBridge()
        self.ld = ColorLineDetector(debug_img_t=LINE_DEBUG_TYPE.RESULT, offset=(0, 0))

    def listener_callback(self, msg):
        """Callback function when message is received"""
        # self.get_logger().info(f'image size {msg.width}x{msg.height}')
        cam_img = np.flipud(msg.data)

        # detect stack and determine pixel-space errors
        _, err, dbg_img = self.ld.detect(cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR))
        print(dbg_img.dtype)

        if self.debug_img: self.img_pub.publish(self.br.cv2_to_imgmsg(dbg_img, encoding="rgb8"))


def main(args=None):
    """Creates subscriber node and spins it"""
    rclpy.init(args=args)

    print(args)

    minimal_subscriber = TowelDetector()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
