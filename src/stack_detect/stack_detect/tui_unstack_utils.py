"""
unstack_utils.py
================
Self-contained unstack logic for use by tui_controller.py.

This is a clean copy of the relevant parts of stack_choice.py with one
key difference: every input() call is replaced by _ask(), which either:

  - shows a TUI modal and waits for Y/N + Enter  (for "continue? (y/N)")
  - auto-confirms silently                        (for "start?", etc.)

stack_choice.py is NOT imported here and is NOT needed at runtime.
tui_controller.py imports only this file.

Public API
----------
  run(state, stack_node, mh2)
      Called from a daemon thread in tui_controller. Runs the full unstack
      sequence (equivalent to stack_choice.main(with_slides=True)).

  StackDetectorDINO
      The ROS node class. Instantiated once in tui_controller.main() and
      passed to run().
"""

import cv2
import time
import rclpy
import numpy as np

from PIL import Image
from cv_bridge import CvBridge

from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
import tf2_ros
from tf2_geometry_msgs import do_transform_point
from tf2_geometry_msgs import PointStamped, PoseStamped
from geometry_msgs.msg import PoseArray, PoseStamped
from std_msgs.msg import Header
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image as ImageMSG, CompressedImage, CameraInfo
from stack_approach.helpers import publish_img, transform_to_pose_stamped
from stack_detect.helpers.dino_model import DINOModel, plot_boxes_to_image
from stack_msgs.srv import StackDetect, RollerGripperV2, RollerGripperV3
from softenable_display_msgs.srv import SetDisplay
from std_srvs.srv import Trigger
from stack_approach.motion_helper_v2 import MotionHelperV2

from scipy.spatial.transform import Rotation as R


# ═══════════════════════════════════════════════════════════════════════════
#  input() routing
#  All input() calls in this file go through _ask(). The ask_fn is set by
#  run() before the sequence starts, so the functions below don't need to
#  know anything about the TUI.
# ═══════════════════════════════════════════════════════════════════════════

# Prompt substrings that require operator confirmation in the TUI modal.
# Everything else is auto-confirmed (returns "").
_CONFIRM_PROMPTS = ("continue? (y/n)",)

# Set by run() at thread start; default passes through to real input()
# so this module is also safe to use standalone if needed.
_ask_fn = input


def _ask(prompt: str = "") -> str:
    return _ask_fn(prompt)


def _needs_confirmation(prompt: str) -> bool:
    return any(p in prompt.strip().lower() for p in _CONFIRM_PROMPTS)


# ═══════════════════════════════════════════════════════════════════════════
#  Geometry helpers  (verbatim from stack_choice.py)
# ═══════════════════════════════════════════════════════════════════════════

def normalize(v):
    return v / np.linalg.norm(v)


def pose_from_axis_mapping(pose, child_x_in_parent=None,
                           child_y_in_parent=None, child_z_in_parent=None):
    axes = {'x': child_x_in_parent, 'y': child_y_in_parent,
            'z': child_z_in_parent}
    for k, v in axes.items():
        if v is not None:
            axes[k] = normalize(np.array(v, dtype=float))
    if axes['x'] is None:
        axes['x'] = normalize(np.cross(axes['y'], axes['z']))
    elif axes['y'] is None:
        axes['y'] = normalize(np.cross(axes['z'], axes['x']))
    elif axes['z'] is None:
        axes['z'] = normalize(np.cross(axes['x'], axes['y']))
    axes['x'] = normalize(axes['x'])
    axes['y'] = normalize(np.cross(axes['z'], axes['x']))
    axes['z'] = normalize(np.cross(axes['x'], axes['y']))
    R_mat = np.column_stack((axes['x'], axes['y'], axes['z']))
    quat = R.from_matrix(R_mat).as_quat()
    pose.pose.orientation.x = float(quat[0])
    pose.pose.orientation.y = float(quat[1])
    pose.pose.orientation.z = float(quat[2])
    pose.pose.orientation.w = float(quat[3])
    return pose


def translate_pose_away_from_line_toward_wrist(pose, pose_a, pose_b,
                                               wrist_tf, distance):
    a = np.array([pose_a.pose.position.x, pose_a.pose.position.y])
    b = np.array([pose_b.pose.position.x, pose_b.pose.position.y])
    p = np.array([pose.pose.position.x, pose.pose.position.y])
    w = np.array([wrist_tf.transform.translation.x,
                  wrist_tf.transform.translation.y])
    line_dir = normalize(b - a)
    perp = np.array([-line_dir[1], line_dir[0]])
    cand1 = p + distance * perp
    cand2 = p - distance * perp
    new_xy = cand1 if np.linalg.norm(cand1 - w) < np.linalg.norm(cand2 - w) \
             else cand2
    pose.pose.position.x = float(new_xy[0])
    pose.pose.position.y = float(new_xy[1])
    return pose


def rotate_about_child_x_toward_line(pose, pose_a, pose_b, child_axis):
    a = np.array([pose_a.pose.position.x, pose_a.pose.position.y])
    b = np.array([pose_b.pose.position.x, pose_b.pose.position.y])
    p = np.array([pose.pose.position.x, pose.pose.position.y])
    ab = b - a
    t = np.dot(p - a, ab) / np.dot(ab, ab)
    closest = a + t * ab
    dir_xy = closest - p
    norm = np.linalg.norm(dir_xy)
    if norm < 1e-6:
        return pose
    dir_xy /= norm
    q = pose.pose.orientation
    rot = R.from_quat([q.x, q.y, q.z, q.w])
    R_mat = rot.as_matrix()
    x_axis = R_mat[:, 0]
    yz_plane_axes = np.array(R_mat[:, 1:3])
    idx = 1 if np.array_equal(child_axis, [0, 1, 0]) else 2
    axis_in_parent = yz_plane_axes[:, idx - 1].copy()
    axis_xy = axis_in_parent[:2]
    axis_xy_norm = np.linalg.norm(axis_xy)
    if axis_xy_norm < 1e-6:
        return pose
    axis_xy /= axis_xy_norm
    dot = np.clip(np.dot(axis_xy, dir_xy), -1.0, 1.0)
    angle = np.arccos(dot)
    cross = axis_xy[0] * dir_xy[1] - axis_xy[1] * dir_xy[0]
    if cross < 0:
        angle = -angle
    yaw_rot = R.from_rotvec(angle * x_axis)
    new_rot = yaw_rot * rot
    new_q = new_rot.as_quat()
    pose.pose.orientation.x = float(new_q[0])
    pose.pose.orientation.y = float(new_q[1])
    pose.pose.orientation.z = float(new_q[2])
    pose.pose.orientation.w = float(new_q[3])
    return pose


def rotate_about_child_axis(pose, axis, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    q = pose.pose.orientation
    rot_current = R.from_quat([q.x, q.y, q.z, q.w])
    axis = np.array(axis, dtype=float)
    axis_in_parent = rot_current.apply(axis)
    rot_delta = R.from_rotvec(angle_rad * axis_in_parent)
    rot_new = rot_delta * rot_current
    q_new = rot_new.as_quat()
    pose.pose.orientation.x = float(q_new[0])
    pose.pose.orientation.y = float(q_new[1])
    pose.pose.orientation.z = float(q_new[2])
    pose.pose.orientation.w = float(q_new[3])
    return pose


def translate_along_child_axis(pose, axis, distance):
    q = pose.pose.orientation
    rot_current = R.from_quat([q.x, q.y, q.z, q.w])
    axis = np.array(axis, dtype=float)
    axis_in_parent = rot_current.apply(axis)
    axis_in_parent /= np.linalg.norm(axis_in_parent)
    pose.pose.position.x += float(axis_in_parent[0] * distance)
    pose.pose.position.y += float(axis_in_parent[1] * distance)
    pose.pose.position.z += float(axis_in_parent[2] * distance)
    return pose


def apply_local_offset(base_pose_stamped, trans_offset=[0., 0., 0.],
                       rot_offset_deg=[0., 0., 0.]):
    p = base_pose_stamped.pose.position
    o = base_pose_stamped.pose.orientation
    base_pos = np.array([p.x, p.y, p.z])
    base_ori = R.from_quat([o.x, o.y, o.z, o.w])
    rotated_trans = base_ori.apply(trans_offset)
    new_pos = base_pos + rotated_trans
    local_rot_offset = R.from_euler('xyz', rot_offset_deg, degrees=True)
    new_ori = base_ori * local_rot_offset
    new_quat = new_ori.as_quat()
    new_pose = PoseStamped()
    new_pose.header = base_pose_stamped.header
    new_pose.pose.position.x = new_pos[0]
    new_pose.pose.position.y = new_pos[1]
    new_pose.pose.position.z = new_pos[2]
    new_pose.pose.orientation.x = new_quat[0]
    new_pose.pose.orientation.y = new_quat[1]
    new_pose.pose.orientation.z = new_quat[2]
    new_pose.pose.orientation.w = new_quat[3]
    return new_pose


# ═══════════════════════════════════════════════════════════════════════════
#  Joint-space constants  (verbatim from stack_choice.py)
# ═══════════════════════════════════════════════════════════════════════════

PRE_STACK_START_THIN = [-2.1758, -1.5987, 2.0216, -1.9133, 2.4286, 3.8264]
SAFE_TRANSITION      = [-1.7321, -0.7593, 1.7561, -2.5065, 1.0454, 3.0375]
POST_STACK           = [-2.2317, -1.5417, 2.0356, -1.9210, 2.4224, 3.9107]

PLACING_SEQUENCE_ALIGNED_V2 = [
    [-1.4939, -0.9910,  2.0959, -1.7361, 1.5754, 4.0280],
    [-1.1825, -0.6518,  1.2790, -1.6061, 0.9424, 4.3434],
    [-0.5799, -1.1447,  1.9910, -1.5407, 1.6626, 3.8611],
    [-0.6099, -1.1205,  2.1281, -1.9246, 1.1741, 3.9345],
    [-1.4473, -0.5199,  2.0482, -2.9699, 0.7932, -0.0090],
]
PLACING_SEQUENCE_ALIGNED_V2_TIMES = [1, 1, 0.8, 0.4, 1.3]


# ═══════════════════════════════════════════════════════════════════════════
#  StackDetectorDINO node  (verbatim from stack_choice.py, with_slides=True)
# ═══════════════════════════════════════════════════════════════════════════

class StackDetectorDINO(Node):

    def __init__(self, with_slides: bool = False):
        super().__init__("StackDetectorDINO")
        self.log = self.get_logger()
        self.with_slides = with_slides

        self.bridge   = CvBridge()
        self.cb_group = ReentrantCallbackGroup()

        self.declare_parameter('cpu_only', False)
        self.cpu_only = self.get_parameter("cpu_only").get_parameter_value().bool_value

        self.data_timeout = 0.1

        self.rgb_img         = None
        self.latest_rgb_time = None
        self.depth_img       = None
        self.latest_depth_time = None
        self.camera_info     = None
        self.latest_info_time = None

        self.img_sub = self.create_subscription(
            CompressedImage, "/camera/color/image_raw/compressed",
            self.rgb_cb, 10, callback_group=self.cb_group)
        self.create_subscription(
            ImageMSG, "/camera/aligned_depth_to_color/image_raw",
            self.depth_cb, 10, callback_group=self.cb_group)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/color/camera_info',
            self.info_cb, 10, callback_group=self.cb_group)

        self.stack_pose_pub = self.create_publisher(
            PoseArray, '/unstack_start_pose', 10, callback_group=self.cb_group)
        self.img_pub = self.create_publisher(
            CompressedImage, '/camera/color/dino/compressed', 0,
            callback_group=self.cb_group)

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info("waiting for tf ...")
        if not self.wait_for_transform(
                "map", "right_arm_l515_color_optical_frame", timeout_sec=10):
            self.get_logger().error("cant tf! bye.")
        self.get_logger().info("got tf!")

        self.get_logger().info("setting up DINO ...")
        self.dino = DINOModel(prefix="/home/ros/pretrained/dino/",
                              cpu_only=self.cpu_only)
        self.get_logger().info("setup done!")

        self.sam_client = self.create_client(StackDetect, "stack_detect")
        while not self.sam_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('SAM service not available, waiting again...')

        if with_slides:
            self.cli_display     = self.create_client(SetDisplay, '/set_display')
            self.cli_start_switch = self.create_client(Trigger, "start_switch")
            self.cli_kill_switch  = self.create_client(Trigger, "kill_switch")
            while not self.cli_display.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('Waiting for /set_display service...')

    def switch_slide(self, slide_name):
        if not self.with_slides:
            return
        self.cli_display.call_async(SetDisplay.Request(name=slide_name))

    def rgb_cb(self, msg):
        try:
            self.rgb_img         = self.bridge.compressed_imgmsg_to_cv2(msg)
            self.latest_rgb_time = time.time()
        except Exception as e:
            self.get_logger().error(f"RGB conversion failed: {e}")

    def depth_cb(self, msg):
        try:
            self.depth_img        = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough")
            self.latest_depth_time = time.time()
        except Exception as e:
            self.get_logger().error(f"Depth conversion failed: {e}")

    def info_cb(self, msg):
        self.camera_info      = msg
        self.latest_info_time = time.time()

    def clear_data(self):
        self.rgb_img = self.depth_img = self.camera_info = None
        self.latest_rgb_time = self.latest_depth_time = self.latest_info_time = None
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def has_fresh_data(self):
        now = time.time()
        if self.rgb_img is None or self.latest_rgb_time is None:
            self.get_logger().warning("no RGB yet ...")
            return False
        if self.depth_img is None or self.latest_depth_time is None:
            self.get_logger().warning("no DEPTH yet ...")
            return False
        if self.camera_info is None or self.latest_info_time is None:
            self.get_logger().warning("no K yet ...")
            return False
        if (now - self.latest_rgb_time)   > self.data_timeout: return False
        if (now - self.latest_depth_time) > self.data_timeout: return False
        if (now - self.latest_info_time)  > self.data_timeout: return False
        self.get_logger().info("waiting for tf ...")
        if not self.wait_for_transform(
                "map", "right_arm_l515_color_optical_frame", timeout_sec=10):
            self.get_logger().error("cant tf! bye.")
        self.get_logger().info("got tf!")
        return True

    def wait_for_transform(self, target_frame, source_frame, timeout_sec=2.0):
        start_time = self.get_clock().now()
        while rclpy.ok():
            if self.tf_buffer.can_transform(
                    target_frame, source_frame, rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.1)):
                return True
            if (self.get_clock().now() - start_time).nanoseconds * 1e-9 > timeout_sec:
                self.get_logger().error(
                    f"Timeout waiting for TF {source_frame} → {target_frame}")
                return False
            rclpy.spin_once(self, timeout_sec=0.1)

    def point_to_map_pose(self, point_stamped):
        if not self.wait_for_transform(
                "map", point_stamped.header.frame_id, timeout_sec=10):
            return None
        try:
            transform = self.tf_buffer.lookup_transform(
                "map", point_stamped.header.frame_id, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5))
            point_in_map = do_transform_point(point_stamped, transform)
        except Exception as e:
            self.get_logger().warning(f"TF transform failed: {e}")
            return None
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp    = self.get_clock().now().to_msg()
        pose.pose.position.x = point_in_map.point.x
        pose.pose.position.y = point_in_map.point.y
        pose.pose.position.z = point_in_map.point.z
        pose.pose.orientation.w = 1.0
        return pose

    def filter_overlapping_boxes(self, boxes, phrases, confidences,
                                 iou_thresh=0.3):
        if len(boxes) == 0:
            return boxes, phrases, confidences
        boxes       = np.array(boxes)
        confidences = np.array(confidences)
        order       = np.argsort(-confidences)
        keep_indices = []
        while len(order) > 0:
            i = order[0]
            keep_indices.append(i)
            rest     = order[1:]
            suppress = []
            x1_i, y1_i, x2_i, y2_i = boxes[i]
            area_i = (x2_i - x1_i) * (y2_i - y1_i)
            for j in rest:
                x1_j, y1_j, x2_j, y2_j = boxes[j]
                area_j   = (x2_j - x1_j) * (y2_j - y1_j)
                xi1 = max(x1_i, x1_j); yi1 = max(y1_i, y1_j)
                xi2 = min(x2_i, x2_j); yi2 = min(y2_i, y2_j)
                inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                if inter_area == 0: continue
                iou = inter_area / (area_i + area_j - inter_area)
                if iou > iou_thresh:
                    suppress.append(j)
            order = np.array([idx for idx in rest if idx not in suppress])
        return (boxes[keep_indices].tolist(),
                [phrases[i] for i in keep_indices],
                confidences[keep_indices].tolist())

    def get_center_point(self, pixel, depth_img, K):
        depth_raw = depth_img[pixel[1], pixel[0]]
        if depth_raw == 0 or np.isnan(depth_raw):
            return None
        return self.pixel_to_point(pixel, float(depth_raw) / 1000.0, K)

    def pixel_to_point(self, pixel, depth, K,
                       frame="right_arm_l515_color_optical_frame"):
        px, py = pixel
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        ps = PointStamped()
        ps.header.frame_id = frame
        ps.header.stamp    = self.get_clock().now().to_msg()
        ps.point.x = float((px - cx) * depth / fx)
        ps.point.y = float((py - cy) * depth / fy)
        ps.point.z = float(depth)
        return ps

    def pre_stack_pose(self):
        while rclpy.ok() and not self.has_fresh_data():
            rclpy.spin_once(self, timeout_sec=0.1)
        tf_map2right_wrist = self.tf_buffer.lookup_transform(
            "map", "right_arm_wrist_3_link", rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=0.5))
        img       = self.rgb_img.copy()
        image_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                    mode="RGB")
        self.get_logger().info("running DINO ...")
        boxes_px, pred_phrases, confidences = self.dino.predict(
            image_pil, "detect all stacks of clothing")
        boxes_px, pred_phrases, confidences = self.filter_overlapping_boxes(
            boxes_px, pred_phrases, confidences, iou_thresh=0.3)
        img_before  = plot_boxes_to_image(
            image_pil.copy(), boxes_px, pred_phrases)[0]
        # With with_slides=True the box is pre-selected; use index 0.
        box_idx = 0
        x1, y1, x2, y2 = boxes_px[box_idx]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        bwidth  = abs(y1 - y2)
        y_right = int(cy - (bwidth * 0.05))
        y_left  = int(cy + (bwidth * 0.05))
        image_with_box = plot_boxes_to_image(
            image_pil.copy(), boxes_px, pred_phrases,
            more_points=[[[cx, y_right], (255, 0, 0)],
                         [[cx, y_left],  (0, 255, 0)]])[0]
        publish_img(self.img_pub, image_with_box)
        K        = np.array(self.camera_info.k).reshape(3, 3)
        pose_map = self.point_to_map_pose(
            self.get_center_point((cx, cy), self.depth_img, K))
        pose_right = self.point_to_map_pose(
            self.get_center_point((cx, y_right), self.depth_img, K))
        pose_left  = self.point_to_map_pose(
            self.get_center_point((cx, y_left),  self.depth_img, K))
        if not pose_map:
            print("ERROR no pose map")
            return
        pose_map = pose_from_axis_mapping(
            pose_map, child_z_in_parent=[0, 1, 0], child_y_in_parent=[1, 0, 0])
        pose_map = translate_pose_away_from_line_toward_wrist(
            pose_map, pose_right, pose_left, tf_map2right_wrist, 0.4)
        pose_map = rotate_about_child_x_toward_line(
            pose_map, pose_right, pose_left, [0, 0, 1])
        pose_map = translate_along_child_axis(pose_map, [0, 1, 0], -0.085)
        pose_map = rotate_about_child_axis(pose_map, [0, 0, 1], 12)
        self.stack_pose_pub.publish(PoseArray(
            header=pose_map.header,
            poses=[pose_map.pose, pose_right.pose, pose_left.pose]))
        return pose_map


# ═══════════════════════════════════════════════════════════════════════════
#  Sequence functions
# ═══════════════════════════════════════════════════════════════════════════

def placing_sequence(mh2: MotionHelperV2):
    scale = 1.0
    for i, (t, q) in enumerate(
            zip(PLACING_SEQUENCE_ALIGNED_V2_TIMES, PLACING_SEQUENCE_ALIGNED_V2)):
        mh2.go_to_q(q, time=scale * t, side="right")
        if i == 2:
            mh2.call_cli_sync(mh2.finger2srv["right_v2"],
                              RollerGripperV2.Request(position=1.0))


def unstack(mh2: MotionHelperV2, node: StackDetectorDINO,
            dist_off=-0.05, height_off=0.021, prim_angle=45):
    has_success = False
    for _ in range(10):
        try:
            fut = node.sam_client.call_async(StackDetect.Request(store_data=True))
            rclpy.spin_until_future_complete(node, fut)
            mh2.call_cli_sync(mh2.finger2srv["right_v2"],
                              RollerGripperV2.Request(position=1.0))
            result = fut.result()
            if not result.success:
                print("SAM2 not successful! trying again ...")
                continue
            stack_pose__cam = result.target_pose
            stack_pose__cam.header.stamp.sec     = 0
            stack_pose__cam.header.stamp.nanosec = 0
            finger__map      = transform_to_pose_stamped(
                node.tf_buffer.lookup_transform(
                    "map", "right_finger", rclpy.time.Time()))
            stack_pose__map  = node.tf_buffer.transform(stack_pose__cam, "map")
            stack_pose__map.pose.orientation = finger__map.pose.orientation
            poses    = []
            dh_offset = 0.003
            for dd, dh in [
                [-0.09,  0.021],
                [-0.035, 0.004],
                [ 0.015, 0.004],
                [ 0.015, 0.015],
                [ 0.040, 0.015],
                [-0.02,  0.015],
            ]:
                p = apply_local_offset(
                    stack_pose__map,
                    trans_offset=[dh + dh_offset, 0, dd],
                    rot_offset_deg=[0, -prim_angle, 0])
                poses.append(mh2.finger_to_wrist(p, "right"))
            node.stack_pose_pub.publish(PoseArray(
                header=stack_pose__map.header,
                poses=[stack_pose__map.pose] + [p.pose for p in poses]))
        except Exception as e:
            print(f"Unstacking exception:\n{e}")

        # ── Only this input() needs operator confirmation ──────────────────
        if _ask("continue? (y/N)").strip().lower() == "y":
            has_success = True
            break

    if not has_success:
        assert False, "unsuccessful unstacking"

    for i, (p, t) in enumerate(zip(poses, [3, 1, 1, .5, .5, .5])):
        mh2.go_to_pose(p, t, side="right", blocking=True)
        time.sleep(0.1)
        if i == len(poses) - 2:
            mh2.call_cli_sync(mh2.finger2srv["right_v3"],
                              RollerGripperV3.Request(effort=0.3))


def _run_sequence(mh2: MotionHelperV2, node: StackDetectorDINO):
    """
    Core sequence – equivalent to stack_choice.main(with_slides=True).
    input("start?") is auto-confirmed by the routing in run(); no call needed.
    """
    mh2.go_to_q(q=SAFE_TRANSITION, time=1, side="right")
    mh2.go_to_q(q=PRE_STACK_START_THIN, time=1, side="right")

    print("RUNNING UNSTACKING!")
    time.sleep(0.7)
    unstack(mh2, node)
    print("unstacking done.")

    mh2.go_to_q(q=POST_STACK, time=1, side="right")
    placing_sequence(mh2)


# ═══════════════════════════════════════════════════════════════════════════
#  Public entry point called from tui_controller
# ═══════════════════════════════════════════════════════════════════════════

def run(state, stack_node: StackDetectorDINO, mh2: MotionHelperV2):
    """
    Entry point called from a daemon thread in tui_controller.

    Installs a routing ask function so that:
      - "continue? (y/N)"  →  TUI modal  (blocks until operator answers)
      - everything else    →  auto-confirmed, logged
    """
    global _ask_fn
    gate = state.unstack_gate

    def routed_ask(prompt=""):
        prompt_str = prompt.strip() if isinstance(prompt, str) else str(prompt)
        if _needs_confirmation(prompt_str):
            state.add_log(f"  [unstack] waiting for confirmation: {prompt_str!r}")
            answer = gate.ask(prompt_str)
            state.add_log(f"  [unstack] got: {answer!r}")
            return answer
        else:
            state.add_log(f"  [unstack] auto-confirmed: {prompt_str!r}")
            return ""

    _ask_fn = routed_ask
    try:
        state.add_log("→ UNSTACK started (with_slides=True)")
        _run_sequence(mh2=mh2, node=stack_node)
        state.add_log("✓ UNSTACK finished")
    except AssertionError as e:
        state.add_log(f"⚠  UNSTACK failed: {e}")
    except Exception as e:
        state.add_log(f"⚠  UNSTACK exception: {e}")
    finally:
        _ask_fn = input          # restore default
        with state._lock:
            state.unstack_running = False
        if gate.waiting:
            gate.answer("n")     # unblock gate if sequence ended unexpectedly