"""Subscriber module"""
import cv2
import rclpy
import numpy as np

from rclpy.node import Node
from cv_bridge import CvBridge 
from rclpy.parameter import Parameter

from std_msgs.msg import Int16MultiArray
from sensor_msgs.msg import Image
from rcl_interfaces.msg import SetParametersResult

from stack_detect.detectors import ColorLineDetector


class TowelDetector(Node):
    """Subscriber node"""

    def __init__(self):
        super().__init__("towel_detector")

        # defaults                                          pink        yellow
        self.declare_parameter("debug_img", "none")
        self.declare_parameter("hue", 48)              #   340             48       
        self.declare_parameter("hue_tol", 0.1)         #    0.04           0.1
        self.declare_parameter("sat_range", [120, 255]) #           [120, 255]
        self.declare_parameter("offset", [0, 0])        #
        self.declare_parameter("morph_size", 4)        #    15             4

        self.debug_img = self.get_parameter("debug_img").value
        self.hue = self.get_parameter("hue").value
        self.hue_tol = self.get_parameter("hue_tol").value
        self.sat_range = self.get_parameter("sat_range").value
        self.offset = self.get_parameter("offset").value
        self.morph_size = self.get_parameter("morph_size").value

        assert self.debug_img in ["final", "all", "none"], 'self.debug_img in ["final", "all", None]'

        self.add_on_set_parameters_callback(self.parameter_callback)

        self.subscription = self.create_subscription(
            Image, "/video_in", self.listener_callback, 0
        )

        self.err_pub = self.create_publisher(Int16MultiArray, 'line_img_error', 10)

        if self.debug_img:
            self.img_pub = self.create_publisher(Image, "debug_img", 10)

        self.br = CvBridge()
        self.ld = ColorLineDetector()
        self.set_detector_params()

        print("towel detection running")

    def set_detector_params(self):
        self.ld.hue = self.hue
        self.ld.hue_tol = self.hue_tol
        self.ld.sat_range = self.sat_range
        self.ld.offset = self.offset
        self.ld.morph_size = self.morph_size

    def parameter_callback(self, params):
        for p in params:
            if hasattr(self, p.name): 
                print(f"setting {p.name} {p.value}")
                setattr(self, p.name, p.value)
        self.set_detector_params()
        return SetParametersResult(successful=True)

    def listener_callback(self, msg):
        """Callback function when message is received"""
        # self.get_logger().info(f'image size {msg.width}x{msg.height}')
        cam_img = self.br.imgmsg_to_cv2(msg, "bgr8")

        # detect stack and determine pixel-space errors
        _, err, (all_img, final_img) = self.ld.detect(cam_img)
        if err is not None: self.err_pub.publish(Int16MultiArray(data=err))

        if self.debug_img:
            dbg_img = all_img if self.debug_img == "all" else final_img

            # cv2 imshow for debugging
            # cv2.imshow("", dbg_img)
            # cv2.waitKey(50)

            self.img_pub.publish(self.br.cv2_to_imgmsg(dbg_img, encoding="bgr8"))

    

def main(args=None):
    """Creates subscriber node and spins it"""
    rclpy.init(args=args)

    towel_detector = TowelDetector()

    rclpy.spin(towel_detector)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    towel_detector.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
