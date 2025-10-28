#!/usr/bin/env python3
import os
import cv2
import math
import numpy as np
from PIL import Image

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile

from cv_bridge import CvBridge
from sensor_msgs.msg import Image as ImageMSG, CompressedImage, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger

# --- helpers ---
from stack_detect.helpers.dino_model import DINOModel
from stack_detect.helpers.bag_detection import expand_bounding_box, get_bag_pose_from_array
from stack_approach.helpers import pixel_to_point


class BagDetectNode(Node):
    def __init__(self):
        super().__init__("bag_detect_dino")
        self.get_logger().info("Initializing BagDetectNode...")

        self.cbg = ReentrantCallbackGroup()
        qos = QoSProfile(depth=1)
        self.bridge = CvBridge()

        # --- Subscribers ---
        self.img_sub = self.create_subscription(
            CompressedImage,
            "/camera/color/image_raw/compressed",
            self.rgb_cb,
            qos,
            callback_group=self.cbg,
        )

        self.depth_sub = self.create_subscription(
            ImageMSG,
            "/camera/aligned_depth_to_color/image_raw",
            self.depth_cb,
            qos,
            callback_group=self.cbg,
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            "/camera/color/camera_info",
            self.info_cb,
            qos,
        )

        # --- Publishers ---
        self.img_pub = self.create_publisher(
            CompressedImage, "/camera/color/bag/compressed", qos, callback_group=self.cbg
        )
        self.bag_pose_pub = self.create_publisher(
            PoseStamped, "/bag_pose", 10, callback_group=self.cbg
        )

        # --- Service ---
        self.trigger_srv = self.create_service(
            Trigger, "detect_bag", self.trigger_cb, callback_group=self.cbg
        )

        # --- State ---
        self.latest_rgb = None
        self.latest_depth = None
        self.K = None  # camera intrinsics
        self.dino = DINOModel(cpu_only=True, prefix=f"/home/ros/pretrained/dino/")

        self.get_logger().info("✅ DINO model loaded and node initialized")

    # -----------------------------------------------------------
    # --- Callbacks --------------------------------------------
    # -----------------------------------------------------------
    def rgb_cb(self, msg: CompressedImage):
        try:
            self.latest_rgb = msg
        except Exception as e:
            self.get_logger().error(f"RGB decode failed: {e}")

    def depth_cb(self, msg: ImageMSG):
        try:
            self.latest_depth = msg
        except Exception as e:
            self.get_logger().error(f"Depth decode failed: {e}")

    def info_cb(self, msg: CameraInfo):
        # Extract intrinsic matrix (3x3)
        self.K = np.array(msg.k, dtype=np.float32).reshape(3, 3)

    # -----------------------------------------------------------
    # --- 2D→3D conversion helper -------------------------------
    # -----------------------------------------------------------
    def pixel_to_3d(self, pixel, depth_img, K):
        """
        Convert a 2D pixel (u,v) + depth image + intrinsics to 3D coordinates (in meters)
        """
        u, v = int(pixel[0]), int(pixel[1])
        if depth_img is None or K is None:
            raise RuntimeError("Missing depth or camera intrinsics")

        if (
            v < 0 or v >= depth_img.shape[0]
            or u < 0 or u >= depth_img.shape[1]
        ):
            raise RuntimeError(f"Pixel {pixel} out of image bounds {depth_img.shape[::-1]}")

        line_dist = depth_img[v, u] / 1000.0  # mm → m
        point_3d = pixel_to_point(pixel, line_dist, K)
        return point_3d

    # -----------------------------------------------------------
    # --- Service trigger ---------------------------------------
    # -----------------------------------------------------------
    def trigger_cb(self, request, response):
        if self.latest_rgb is None:
            response.success = False
            response.message = "No RGB image received yet."
            return response

        if self.latest_depth is None:
            response.success = False
            response.message = "No depth image received yet."
            return response

        if self.K is None:
            response.success = False
            response.message = "No camera info received yet."
            return response

        try:
            image_cv = self.bridge.compressed_imgmsg_to_cv2(self.latest_rgb)
            image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
            boxes, labels, confidences = self.dino.predict(image_pil, "bag")

            if len(boxes) == 0:
                response.success = False
                response.message = "No bag detected."
                return response

            # --- Crop ---
            best_box = expand_bounding_box(boxes[0], image_pil.width, image_pil.height, scale=1.15)
            x0, y0, x1, y1 = best_box
            cropped_cv = image_cv[y0:y1, x0:x1]

            # --- Pose estimation ---
            angle, box, offset_point = get_bag_pose_from_array(cropped_cv, point_offset=0.15)
            box_global = box + np.array([x0, y0])
            offset_global = (offset_point[0] + x0, offset_point[1] + y0)

            # --- 2D → 3D conversion ---
            point_3d = self.pixel_to_3d(
                offset_global, 
                self.bridge.imgmsg_to_cv2(self.latest_depth), 
                self.K
            )

            # --- Draw result ---
            vis = image_cv.copy()
            cv2.drawContours(vis, [box_global.astype(int)], 0, (0, 0, 255), 2)
            cv2.circle(vis, offset_global, 6, (0, 255, 255), -1)
            cv2.putText(vis, f"{angle:.2f}°", (int(offset_global[0]) + 10, int(offset_global[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # --- Publish annotated image ---
            img_msg = self.bridge.cv2_to_compressed_imgmsg(vis)
            self.img_pub.publish(img_msg)

            # --- Publish 3D pose ---
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = "camera_link"
            pose_msg.pose.position.x = float(point_3d[0])
            pose_msg.pose.position.y = float(point_3d[1])
            pose_msg.pose.position.z = float(point_3d[2])
            pose_msg.pose.orientation.w = 1.0
            self.bag_pose_pub.publish(pose_msg)

            response.success = True
            response.message = (
                f"Bag detected. Angle={angle:.2f}°, Pose=({point_3d[0]:.3f}, {point_3d[1]:.3f}, {point_3d[2]:.3f})"
            )
            self.get_logger().info(response.message)

        except Exception as e:
            self.get_logger().error(f"Detection failed: {e}")
            response.success = False
            response.message = str(e)

        return response


def main(args=None):
    rclpy.init(args=args)
    node = BagDetectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
