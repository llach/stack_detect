#!/usr/bin/env python3
import cv2
import time
import numpy as np
from PIL import Image

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor

from cv_bridge import CvBridge
from sensor_msgs.msg import Image as ImageMSG, CompressedImage, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger

import tf2_ros
from tf2_ros import TransformException
from tf2_geometry_msgs import do_transform_pose

# --- helpers ---
from stack_detect.helpers.dino_model import DINOModel
from stack_detect.helpers.bag_detection import expand_bounding_box, get_bag_pose_from_array
from stack_approach.helpers import pixel_to_point

from tf_transformations import quaternion_from_euler, quaternion_multiply
from geometry_msgs.msg import Quaternion

def rpy_to_quat(roll, pitch, yaw):
    q = quaternion_from_euler(roll, pitch, yaw)  # returns [x, y, z, w]
    quat = Quaternion()
    quat.x, quat.y, quat.z, quat.w = q
    return quat

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
        self.wrist_pose_pub = self.create_publisher(
            PoseStamped, "/bag_wrist_pose", 10, callback_group=self.cbg
        )

        # --- TF listener ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- State ---
        self.latest_rgb = None
        self.latest_depth = None
        self.K = None  # camera intrinsics
        self.dino = DINOModel(cpu_only=True, prefix=f"/home/ros/pretrained/dino/")

        self.get_logger().info("waiting for transforms ...")
        while not (
            self.tf_buffer.can_transform("map", "right_arm_l515_color_optical_frame", rclpy.time.Time()) and 
            self.tf_buffer.can_transform("map", "right_arm_wrist_3_link", rclpy.time.Time()) and 
            self.tf_buffer.can_transform("map", "left_arm_wrist_3_link", rclpy.time.Time()) 
        ):
            time.sleep(0.05)
            rclpy.spin_once(self)

        # --- Service ---
        self.trigger_srv = self.create_service(
            Trigger, "detect_bag", self.trigger_cb, callback_group=self.cbg
        )

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
        self.K = np.array(msg.k, dtype=np.float32).reshape(3, 3)

    # -----------------------------------------------------------
    # --- 2D→3D conversion helper -------------------------------
    # -----------------------------------------------------------
    def pixel_to_3d(self, pixel, depth_img, K):
        """Convert 2D pixel (u,v) + depth image + intrinsics to 3D camera coordinates."""
        u, v = int(pixel[0]), int(pixel[1])
        if depth_img is None or K is None:
            raise RuntimeError("Missing depth or camera intrinsics")

        if (
            v < 0 or v >= depth_img.shape[0]
            or u < 0 or u >= depth_img.shape[1]
        ):
            raise RuntimeError(f"Pixel {pixel} out of bounds {depth_img.shape[::-1]}")

        line_dist = depth_img[v, u] / 1000.0  # mm → m
        point_3d = pixel_to_point(pixel, line_dist, K)
        return point_3d

    # -----------------------------------------------------------
    # --- Transform pose to map frame ---------------------------
    # -----------------------------------------------------------
    def get_bag_pose_in_map(self, pose_cam_frame: PoseStamped) -> PoseStamped:
        """Transform PoseStamped from camera_link to map frame."""
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame="map",
                source_frame=pose_cam_frame.header.frame_id,
                time=rclpy.time.Time(seconds=0),
                # timeout=Duration(seconds=5.0),
            )

            print(f"##### TF time diff {self.get_clock().now().to_msg().sec - transform.header.stamp.sec}")

            pose_map = PoseStamped()
            pose_map.header.frame_id = "map"
            pose_map.header.stamp = self.get_clock().now().to_msg()
            pose_map.pose = do_transform_pose(pose_cam_frame.pose, transform)

            pose_map.pose.orientation.x = 0.0
            pose_map.pose.orientation.y = 0.0
            pose_map.pose.orientation.z = 0.0
            pose_map.pose.orientation.w = 1.0

            return pose_map
        except TransformException as ex:
            self.get_logger().warn(f"TF transform failed: {ex}")
            return None     

    def get_left_wrist_pose(self, angle_deg: float) -> PoseStamped:
        """Return PoseStamped of left wrist rotated around its local Z axis by `angle_deg`."""
        # --- Base pose (map frame) ---
        base_pos = np.array([0.468, -0.288, 0.940])
        base_quat = np.array([-0.619, 0.785, 0.009, -0.007])  # (x, y, z, w)

        # --- Rotation around local Z (in that frame) ---
        rot_local_z = quaternion_from_euler(0.0, 0.0, np.deg2rad(angle_deg))  # rotation about Z

        # Multiply quaternions: new_q = base_q * local_z_rot
        # (order matters: local rotation applied in local frame)
        new_quat = quaternion_multiply(base_quat, rot_local_z)

        # --- Compose PoseStamped ---
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = base_pos[0]
        pose.pose.position.y = base_pos[1]
        pose.pose.position.z = base_pos[2]
        pose.pose.orientation.x = new_quat[0]
        pose.pose.orientation.y = new_quat[1]
        pose.pose.orientation.z = new_quat[2]
        pose.pose.orientation.w = new_quat[3]

        return pose

    # -----------------------------------------------------------
    # --- Trigger Service: main pipeline ------------------------
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

        # --- Decode latest images ---
        image_cv = self.bridge.compressed_imgmsg_to_cv2(self.latest_rgb)
        depth_cv = self.bridge.imgmsg_to_cv2(self.latest_depth, desired_encoding="passthrough")
        image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

        # --- DINO detection ---
        boxes, labels, confidences = self.dino.predict(image_pil, "bag")
        if len(boxes) == 0:
            response.success = False
            response.message = "No bag detected."
            return response

        # --- Crop around best detection ---
        best_box = expand_bounding_box(boxes[0], image_pil.width, image_pil.height, scale=1.15)
        x0, y0, x1, y1 = best_box
        cropped_cv = image_cv[y0:y1, x0:x1]

        # --- Get bag orientation & offset point ---
        angle, box, offset_point, contour = get_bag_pose_from_array(cropped_cv, point_offset=0.15)
        box_global = box + np.array([x0, y0])
        offset_global = (offset_point[0] + x0, offset_point[1] + y0)

        ## APPLY OFFSET TO ANGLE 
        if angle > 50:
            angle -= 90

        # --- Convert offset pixel → 3D point ---
        point_3d = self.pixel_to_3d(offset_global, depth_cv, self.K)

        # --- Draw visualization ---
        vis = image_cv.copy()

        # 1️⃣ Draw the expanded DINO bounding box
        cv2.rectangle(
            vis,
            (int(x0), int(y0)),
            (int(x1), int(y1)),
            (0, 255, 0), 2
        )
        cv2.putText(vis, "Expanded DINO BB", (x0, max(0, y0 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 2️⃣ Draw the detected contour (adjusted to global coordinates)
        if contour is not None and len(contour) > 0:
            contour_global = contour + np.array([[x0, y0]])
            cv2.drawContours(vis, [contour_global.astype(int)], -1, (255, 0, 0), 2)
            cv2.putText(vis, "Contour", (int(contour_global[0][0][0]), int(contour_global[0][0][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # 3️⃣ Draw the rotated rectangle (bag pose box)
        cv2.drawContours(vis, [box_global.astype(int)], 0, (0, 0, 255), 2)

        # 4️⃣ Draw the offset point + angle text
        cv2.circle(vis, offset_global, 6, (0, 255, 255), -1)
        cv2.putText(vis, f"{angle:.2f}°", (int(offset_global[0]) + 10, int(offset_global[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # --- Publish annotated image ---
        img_msg = self.bridge.cv2_to_compressed_imgmsg(vis)
        self.img_pub.publish(img_msg)

        # --- Compose pose in camera_link ---
        pose_cam = PoseStamped()
        pose_cam.header.stamp = self.get_clock().now().to_msg()
        pose_cam.header.frame_id = "right_arm_l515_color_optical_frame"
        pose_cam.pose.position.x = float(point_3d.point.x)
        pose_cam.pose.position.y = float(point_3d.point.y)
        pose_cam.pose.position.z = float(point_3d.point.z)
        pose_cam.pose.orientation.w = 1.0

        # --- Transform to map frame ---
        pose_map = self.get_bag_pose_in_map(pose_cam)
        if not pose_map:
            response.success = False
            response.message = "tf error"

        pose_map.pose.orientation = rpy_to_quat(0,0,np.deg2rad(-angle))
        self.bag_pose_pub.publish(pose_map)

        wrist_pose = self.get_left_wrist_pose(angle)
        self.wrist_pose_pub.publish(wrist_pose)

        response.success = True
        response.message = (
            f"Bag detected. Angle={angle:.2f}°, MapPose=({pose_map.pose.position.x:.3f}, "
            f"{pose_map.pose.position.y:.3f}, {pose_map.pose.position.z:.3f})"
        )

    

        self.get_logger().info(response.message)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = BagDetectNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
