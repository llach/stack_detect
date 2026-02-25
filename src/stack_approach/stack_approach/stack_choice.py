#!/usr/bin/env python3
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
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Header
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image as ImageMSG, CompressedImage, CameraInfo
from stack_approach.helpers import publish_img, transform_to_pose_stamped
from stack_detect.helpers.dino_model import DINOModel, plot_boxes_to_image
from stack_msgs.srv import StackDetect, RollerGripperV2


from motion_helper_v2 import MotionHelperV2

import numpy as np
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PoseStamped


def normalize(v):
    return v / np.linalg.norm(v)


def pose_from_axis_mapping(
    pose: PoseStamped,
    child_x_in_parent=None,
    child_y_in_parent=None,
    child_z_in_parent=None
):
    """
    Set pose.pose.orientation by specifying where child axes point
    in the parent frame. Provide any two axes; the third is computed.
    """

    axes = {
        'x': child_x_in_parent,
        'y': child_y_in_parent,
        'z': child_z_in_parent
    }

    # Normalize provided axes
    for k, v in axes.items():
        if v is not None:
            axes[k] = normalize(np.array(v, dtype=float))

    # Compute missing axis using right-hand rule
    if axes['x'] is None:
        axes['x'] = normalize(np.cross(axes['y'], axes['z']))
    elif axes['y'] is None:
        axes['y'] = normalize(np.cross(axes['z'], axes['x']))
    elif axes['z'] is None:
        axes['z'] = normalize(np.cross(axes['x'], axes['y']))

    # Re-orthogonalize for safety
    axes['x'] = normalize(axes['x'])
    axes['y'] = normalize(np.cross(axes['z'], axes['x']))
    axes['z'] = normalize(np.cross(axes['x'], axes['y']))

    # Rotation matrix: columns = child axes in parent frame
    R_mat = np.column_stack((axes['x'], axes['y'], axes['z']))

    # Convert to quaternion (x, y, z, w)
    quat = R.from_matrix(R_mat).as_quat()

    pose.pose.orientation.x = float(quat[0])
    pose.pose.orientation.y = float(quat[1])
    pose.pose.orientation.z = float(quat[2])
    pose.pose.orientation.w = float(quat[3])

    return pose

def translate_pose_away_from_line_toward_wrist(
    pose: PoseStamped,
    pose_a: PoseStamped,
    pose_b: PoseStamped,
    wrist_tf,          # TransformStamped (map → wrist)
    distance: float
):
    # --- Extract XY points ---
    a = np.array([pose_a.pose.position.x, pose_a.pose.position.y])
    b = np.array([pose_b.pose.position.x, pose_b.pose.position.y])
    p = np.array([pose.pose.position.x, pose.pose.position.y])
    w = np.array([
        wrist_tf.transform.translation.x,
        wrist_tf.transform.translation.y
    ])

    # --- Line direction ---
    line_dir = normalize(b - a)

    # --- Perpendicular in XY ---
    perp = np.array([-line_dir[1], line_dir[0]])

    # --- Two candidate positions ---
    cand1 = p + distance * perp
    cand2 = p - distance * perp

    # --- Choose one closer to wrist ---
    if np.linalg.norm(cand1 - w) < np.linalg.norm(cand2 - w):
        new_xy = cand1
    else:
        new_xy = cand2

    # --- Apply translation (Z unchanged) ---
    pose.pose.position.x = float(new_xy[0])
    pose.pose.position.y = float(new_xy[1])

    return pose

import numpy as np
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PoseStamped

from softenable_display_msgs.srv import SetDisplay
from std_srvs.srv import Trigger

def rotate_about_child_x_toward_line(
    pose: PoseStamped,
    pose_a: PoseStamped,
    pose_b: PoseStamped,
    child_axis: list,   # which axis of child to point to line, e.g., [0,0,1]
):
    """
    Rotates pose about its own X axis so that child_axis (in child frame) points
    toward the line defined by pose_a and pose_b.
    Assumes child YZ plane is parallel to parent XY plane, X vertical.
    """
    # --- Positions in XY plane ---
    a = np.array([pose_a.pose.position.x, pose_a.pose.position.y])
    b = np.array([pose_b.pose.position.x, pose_b.pose.position.y])
    p = np.array([pose.pose.position.x, pose.pose.position.y])

    # --- Closest point on line AB ---
    ab = b - a
    t = np.dot(p - a, ab) / np.dot(ab, ab)
    closest = a + t * ab

    # --- Desired direction in XY ---
    dir_xy = closest - p
    norm = np.linalg.norm(dir_xy)
    if norm < 1e-6:
        return pose  # already on line
    dir_xy /= norm

    # --- Current rotation matrix ---
    q = pose.pose.orientation
    rot = R.from_quat([q.x, q.y, q.z, q.w])
    R_mat = rot.as_matrix()

    # Child X axis is vertical; YZ plane is horizontal
    x_axis = R_mat[:,0]
    yz_plane_axes = np.array(R_mat[:,1:3])  # columns = child Y, child Z

    # Transform chosen child axis into parent frame
    idx = 1 if np.array_equal(child_axis, [0,1,0]) else 2  # Y=1, Z=2
    axis_in_parent = yz_plane_axes[:, idx-1].copy()

    # Project into XY plane
    axis_xy = axis_in_parent[:2]
    axis_xy_norm = np.linalg.norm(axis_xy)
    if axis_xy_norm < 1e-6:
        return pose  # axis vertical, nothing to do
    axis_xy /= axis_xy_norm

    # --- Compute signed angle between current axis_xy and desired dir_xy ---
    dot = np.clip(np.dot(axis_xy, dir_xy), -1.0, 1.0)
    angle = np.arccos(dot)
    cross = axis_xy[0]*dir_xy[1] - axis_xy[1]*dir_xy[0]
    if cross < 0:
        angle = -angle

    # --- Rotation about child X ---
    yaw_rot = R.from_rotvec(angle * x_axis)  # axis-angle: angle around child X
    new_rot = yaw_rot * rot
    new_q = new_rot.as_quat()

    pose.pose.orientation.x = float(new_q[0])
    pose.pose.orientation.y = float(new_q[1])
    pose.pose.orientation.z = float(new_q[2])
    pose.pose.orientation.w = float(new_q[3])

    return pose

def rotate_about_child_axis(pose: PoseStamped, axis: list, angle_deg: float) -> PoseStamped:
    """
    Rotate the pose about a specified axis in its own frame.

    :param pose: PoseStamped to rotate
    :param axis: 3-element list, axis in child frame (e.g., [0,0,1])
    :param angle_deg: rotation angle in degrees
    :return: PoseStamped with updated orientation
    """
    # Convert angle to radians
    angle_rad = np.deg2rad(angle_deg)

    # Current rotation as scipy Rotation
    q = pose.pose.orientation
    rot_current = R.from_quat([q.x, q.y, q.z, q.w])

    # Axis in child frame → multiply by current rotation to get axis in parent frame
    axis = np.array(axis, dtype=float)
    axis_in_parent = rot_current.apply(axis)  # rotates axis vector into parent frame

    # Rotation about axis
    rot_delta = R.from_rotvec(angle_rad * axis_in_parent)

    # New rotation
    rot_new = rot_delta * rot_current
    q_new = rot_new.as_quat()

    # Update pose
    pose.pose.orientation.x = float(q_new[0])
    pose.pose.orientation.y = float(q_new[1])
    pose.pose.orientation.z = float(q_new[2])
    pose.pose.orientation.w = float(q_new[3])

    return pose

def translate_along_child_axis(pose: PoseStamped, axis: list, distance: float) -> PoseStamped:
    """
    Translate the pose along a given axis in the child frame.

    :param pose: PoseStamped to translate
    :param axis: 3-element list, axis in child frame (e.g., [1,0,0])
    :param distance: translation distance along that axis (meters)
    :return: PoseStamped with updated position
    """
    # Current rotation
    q = pose.pose.orientation
    rot_current = R.from_quat([q.x, q.y, q.z, q.w])

    # Convert child axis to parent frame
    axis = np.array(axis, dtype=float)
    axis_in_parent = rot_current.apply(axis)

    # Normalize in case axis was not unit
    axis_in_parent /= np.linalg.norm(axis_in_parent)

    # Apply translation
    pose.pose.position.x += float(axis_in_parent[0] * distance)
    pose.pose.position.y += float(axis_in_parent[1] * distance)
    pose.pose.position.z += float(axis_in_parent[2] * distance)

    return pose

def get_box_click_cv2(image_pil, boxes_px):
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    # Using a list for a mutable reference inside the nested function
    clicked_index = [-1]

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, (x1, y1, x2, y2) in enumerate(boxes_px):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    clicked_index[0] = i
                    break

    win_name = "Click on a box"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win_name, mouse_callback)

    print("Select a box in the OpenCV window. Press 'q' to cancel.")

    while True:
        cv2.imshow(win_name, image_cv)
        key = cv2.waitKey(1) & 0xFF
        
        # Break if index is selected or user presses 'q'
        if clicked_index[0] != -1 or key == ord('q'):
            break
        
        # Check if window was closed via the [X] button
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyWindow(win_name)
    # Crucial for some OS backends to actually close the window
    for _ in range(10):
        cv2.waitKey(1)
        
    return clicked_index[0]

class StackDetectorDINO(Node):

    def __init__(self, with_slides: bool = False):
        super().__init__("StackDetectorDINO")
        self.log = self.get_logger()
        self.with_slides = with_slides

        self.bridge = CvBridge()
        self.cb_group = ReentrantCallbackGroup()

        self.declare_parameter('cpu_only', False)
        self.cpu_only = self.get_parameter("cpu_only").get_parameter_value().bool_value
        
        self.data_timeout = 0.1  # seconds

        # RGB
        self.rgb_img = None
        self.latest_rgb_time = None

        # Depth
        self.depth_img = None
        self.latest_depth_time = None

        # Camera Info
        self.camera_info = None
        self.latest_info_time = None


        self.img_sub = self.create_subscription(
            CompressedImage, "/camera/color/image_raw/compressed", self.rgb_cb, 10, callback_group=self.cb_group
        )

        self.create_subscription(
            ImageMSG, "/camera/aligned_depth_to_color/image_raw", self.depth_cb, 10, callback_group=self.cb_group
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/color/camera_info', self.info_cb, 10, callback_group=self.cb_group
        )

        self.stack_pose_pub = self.create_publisher(PoseArray, '/unstack_start_pose', 10, callback_group=self.cb_group)
        self.img_pub = self.create_publisher(CompressedImage, '/camera/color/dino/compressed', 0, callback_group=self.cb_group)


        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info("waiting for tf ...")
        if not self.wait_for_transform("map", "right_arm_l515_color_optical_frame", timeout_sec=10):
            self.get_logger().error("cant tf! bye.")
        self.get_logger().info("got tf!")

        ### DINO setup
        self.get_logger().info("setting up DINO ...")
        self.dino = DINOModel(prefix="/home/ros/pretrained/dino/", cpu_only=self.cpu_only)
        self.get_logger().info("setup done!")

        self.sam_client = self.create_client(StackDetect, "stack_detect")
        while not self.sam_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('SAM service not available, waiting again...')

        if with_slides:
            self.cli_display = self.create_client(SetDisplay, '/set_display')
            self.cli_start_switch = self.create_client(Trigger, "start_switch")
            self.cli_kill_switch = self.create_client(Trigger, "kill_switch")

            while not self.cli_display.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('Waiting for /set_display service...')

    def switch_slide(self, slide_name):
        if not self.with_slides:
            print("NOT switching slides!")
            return
        
        print(f"switching to slide {slide_name}")
        self.cli_display.call_async(SetDisplay.Request(name=slide_name))

    def rgb_cb(self, msg: CompressedImage):
        try:
            self.rgb_img = self.bridge.compressed_imgmsg_to_cv2(msg)
            self.latest_rgb_time = time.time()
        except Exception as e:
            self.get_logger().error(f"RGB conversion failed: {e}")

    def depth_cb(self, msg: ImageMSG):
        try:
            self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.latest_depth_time = time.time()
        except Exception as e:
            self.get_logger().error(f"Depth conversion failed: {e}")

    def info_cb(self, msg: CameraInfo):
        self.camera_info = msg
        self.latest_info_time = time.time()

    def clear_data(self):
        self.rgb_img, self.depth_img, self.camera_info = None, None, None
        self.latest_rgb_time, self.latest_depth_time, self.latest_info_time = None, None, None

        self.tf_buffer = tf2_ros.Buffer()
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

        if (now - self.latest_rgb_time) > self.data_timeout:
            return False
        if (now - self.latest_depth_time) > self.data_timeout:
            return False
        if (now - self.latest_info_time) > self.data_timeout:
            return False
        
        self.get_logger().info("waiting for tf ...")
        if not self.wait_for_transform("map", "right_arm_l515_color_optical_frame", timeout_sec=10):
            self.get_logger().error("cant tf! bye.")
        self.get_logger().info("got tf!")

        return True
    
    def wait_for_transform(self, target_frame, source_frame, timeout_sec=2.0):
        start_time = self.get_clock().now()

        while rclpy.ok():
            if self.tf_buffer.can_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            ):
                return True

            if (self.get_clock().now() - start_time).nanoseconds * 1e-9 > timeout_sec:
                self.get_logger().error(
                    f"Timeout waiting for TF {source_frame} → {target_frame}"
                )
                return False

            rclpy.spin_once(self, timeout_sec=0.1)

    def point_to_map_pose(self, point_stamped):
        if not self.wait_for_transform("map", point_stamped.header.frame_id, timeout_sec=10):
            return None

        try:
            transform = self.tf_buffer.lookup_transform(
                "map",
                point_stamped.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5)
            )

            point_in_map = do_transform_point(point_stamped, transform)

        except Exception as e:
            self.get_logger().warning(f"TF transform failed: {e}")
            return None

        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = point_in_map.point.x
        pose.pose.position.y = point_in_map.point.y
        pose.pose.position.z = point_in_map.point.z

        # Neutral orientation (facing forward)
        pose.pose.orientation.w = 1.0

        return pose


    def filter_overlapping_boxes(self, boxes, phrases, confidences, iou_thresh=0.3):
        if len(boxes) == 0:
            return boxes, phrases, confidences

        boxes = np.array(boxes)
        confidences = np.array(confidences)

        # Sort by confidence descending
        order = np.argsort(-confidences)

        keep_indices = []

        while len(order) > 0:
            i = order[0]
            keep_indices.append(i)

            rest = order[1:]
            suppress = []

            x1_i, y1_i, x2_i, y2_i = boxes[i]
            area_i = (x2_i - x1_i) * (y2_i - y1_i)

            for j in rest:
                x1_j, y1_j, x2_j, y2_j = boxes[j]
                area_j = (x2_j - x1_j) * (y2_j - y1_j)

                # Intersection
                xi1 = max(x1_i, x1_j)
                yi1 = max(y1_i, y1_j)
                xi2 = min(x2_i, x2_j)
                yi2 = min(y2_i, y2_j)

                inter_w = max(0, xi2 - xi1)
                inter_h = max(0, yi2 - yi1)
                inter_area = inter_w * inter_h

                if inter_area == 0:
                    continue

                union_area = area_i + area_j - inter_area
                iou = inter_area / union_area

                if iou > iou_thresh:
                    suppress.append(j)

            # Remove suppressed indices from further consideration
            order = np.array([idx for idx in rest if idx not in suppress])

        filtered_boxes = boxes[keep_indices].tolist()
        filtered_phrases = [phrases[i] for i in keep_indices]
        filtered_confidences = confidences[keep_indices].tolist()

        return filtered_boxes, filtered_phrases, filtered_confidences

    def get_center_point(self, pixel, depth_img, K):
        depth_raw = depth_img[pixel[1], pixel[0]]

        if depth_raw == 0 or np.isnan(depth_raw):
            return None  # invalid depth

        depth_m = float(depth_raw) / 1000.0  # mm → meters
        return self.pixel_to_point(pixel, depth_m, K)

    def pixel_to_point(self, pixel, depth, K, frame="right_arm_l515_color_optical_frame"):
        px, py = pixel
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        x = (px - cx) * depth / fx
        y = (py - cy) * depth / fy
        z = depth

        ps = PointStamped()
        ps.header.frame_id = frame
        ps.header.stamp = self.get_clock().now().to_msg()

        ps.point.x = float(x)
        ps.point.y = float(y)
        ps.point.z = float(z)

        return ps

    def pre_stack_pose(self):
        while rclpy.ok() and not self.has_fresh_data():
            rclpy.spin_once(self, timeout_sec=0.1)

        tf_map2right_wrist = self.tf_buffer.lookup_transform(
            "map",
            "right_arm_wrist_3_link",
            rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=0.5)
        )
        
        img = self.rgb_img.copy()
        image_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), mode="RGB")

        # ----- Run DINO -----
        dino_start = time.time()
        self.get_logger().info("running DINO ...")

        boxes_px, pred_phrases, confidences = self.dino.predict(
            image_pil,
            "detect all stacks of clothing"
        )

        boxes_px, pred_phrases, confidences = self.filter_overlapping_boxes(
            boxes_px, pred_phrases, confidences, iou_thresh=0.3
        )
        self.get_logger().info(f"DINO took {round(time.time()-dino_start, 2)}s")

        img_before = plot_boxes_to_image(image_pil.copy(), boxes_px, pred_phrases)[0]
        box_idx = get_box_click_cv2(img_before, boxes_px)
        if box_idx < 0:
            print("ERROR picking box idx. bye.")
            return
        print(f"GRASPING BOX IDX {box_idx}")

        # Center pixel of box
        x1, y1, x2, y2 = boxes_px[box_idx]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        bwidth = abs(y1-y2)
        y_right = int(cy - (bwidth * 0.05))
        y_left = int(cy + (bwidth * 0.05))

        image_with_box = plot_boxes_to_image(image_pil.copy(), boxes_px, pred_phrases, more_points=[[[cx,y_right], (255,0,0)],[[cx,y_left], (0,255,0)]])[0]
        publish_img(self.img_pub, image_with_box)

        K = np.array(self.camera_info.k).reshape(3, 3)
        pose_map = self.point_to_map_pose(self.get_center_point((cx, cy), self.depth_img, K))
        pose_right = self.point_to_map_pose(self.get_center_point((cx, y_right), self.depth_img, K))
        pose_left = self.point_to_map_pose(self.get_center_point((cx, y_left), self.depth_img, K))
        
        if not pose_map:
            print("ERROR no pose map")
            return

        pose_map = pose_from_axis_mapping(
            pose_map,
            child_z_in_parent=[0, 1, 0],  # parent Y
            child_y_in_parent=[1, 0, 0]   # parent X
        )
        pose_map = translate_pose_away_from_line_toward_wrist(pose_map, pose_right, pose_left, tf_map2right_wrist, 0.4)
        pose_map = rotate_about_child_x_toward_line(pose_map, pose_right, pose_left, [0,0,1])
        pose_map = translate_along_child_axis(pose_map, [0,1,0], -0.085)
        pose_map = rotate_about_child_axis(pose_map, [0,0,1], 12)

        self.stack_pose_pub.publish(PoseArray(header=pose_map.header, poses=[pose_map.pose, pose_right.pose, pose_left.pose]))

        return pose_map


def start_pose_and_stack_choice(node: StackDetectorDINO):
    node.clear_data()

    # Spin until we have fresh data
    print("getting fresh data ...")
    while rclpy.ok() and not node.has_fresh_data():
        rclpy.spin_once(node, timeout_sec=0.1)
    print("done!")

    node.get_logger().info("Got fresh data — running detection once.")
    pose = node.pre_stack_pose()

    # Optional: keep spinning so image publisher can flush
    for _ in range(5):
        rclpy.spin_once(node, timeout_sec=0.1)

    return pose


from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
import numpy as np

def apply_local_offset(base_pose_stamped, trans_offset=[0.0, 0.0, 0.0], rot_offset_deg=[0.0, 0.0, 0.0]):
    """
    Applies a translation then a rotation offset in the local frame of the given pose.
    :param base_pose_stamped: PoseStamped message (the starting point)
    :param trans_offset: list/array [x, y, z] in meters
    :param rot_offset_deg: list/array [r, p, y] in degrees
    """
    # 1. Extract base position and orientation
    p = base_pose_stamped.pose.position
    o = base_pose_stamped.pose.orientation
    
    base_pos = np.array([p.x, p.y, p.z])
    base_ori = R.from_quat([o.x, o.y, o.z, o.w])

    # 2. Apply Translation in local frame
    # We rotate the local translation vector by the base orientation 
    # to find the movement vector in the global (map) frame.
    rotated_trans = base_ori.apply(trans_offset)
    new_pos = base_pos + rotated_trans

    # 3. Apply Rotation in local frame
    # We create the offset rotation and post-multiply it 
    # (Local rotations happen on the right)
    local_rot_offset = R.from_euler('xyz', rot_offset_deg, degrees=True)
    new_ori = base_ori * local_rot_offset
    new_quat = new_ori.as_quat()

    # 4. Construct the new PoseStamped
    new_pose = PoseStamped()
    new_pose.header = base_pose_stamped.header # Maintain same frame_id and timestamp
    
    new_pose.pose.position.x = new_pos[0]
    new_pose.pose.position.y = new_pos[1]
    new_pose.pose.position.z = new_pos[2]
    
    new_pose.pose.orientation.x = new_quat[0]
    new_pose.pose.orientation.y = new_quat[1]
    new_pose.pose.orientation.z = new_quat[2]
    new_pose.pose.orientation.w = new_quat[3]

    return new_pose

def unstack(mh2: MotionHelperV2, node, dist_off=-0.05, height_off=0.021, prim_angle=45):
    for _ in range(10):
        fut = node.sam_client.call_async(StackDetect.Request())
        rclpy.spin_until_future_complete(node, fut)

        mh2.call_cli_sync(mh2.finger2srv["right_v2"], RollerGripperV2.Request(position=1.0))

        stack_pose__cam = fut.result().target_pose
        stack_pose__cam.header.stamp.sec = 0
        stack_pose__cam.header.stamp.nanosec = 0

        finger__map = transform_to_pose_stamped(node.tf_buffer.lookup_transform("map", "right_finger", rclpy.time.Time()))

        stack_pose__map = node.tf_buffer.transform(stack_pose__cam, "map")
        stack_pose__map.pose.orientation = finger__map.pose.orientation

        poses = []
        for dd, dh in [
            [-0.06, 0.021],     # start
            [-0.035, 0.004],    # pre-stack
            [0.015, 0.004],    # insert
            [0.015, 0.015],     # lift
            [0.040, 0.015],     # further insertion
            [-0.02, 0.015],     # pull out
        ]:
            p = apply_local_offset(stack_pose__map, trans_offset=[dh, 0, dd], rot_offset_deg=[0,-prim_angle,0])
            poses.append(mh2.finger_to_wrist(p, "right"))

        node.stack_pose_pub.publish(PoseArray(header=stack_pose__map.header, poses=[stack_pose__map.pose]+[p.pose for p in poses]))
        print("done ...")
        if input("continue? (y/N)").strip().lower() == "y": break

    for i, (p, t) in enumerate(zip(poses, [
        3,      # start
        1,      # pre-stack
        1,      # insert
        .5,     # lift
        .5,     # further insertion
        .5,     # pull out
    ])):
        mh2.go_to_pose(p, t, side="right", blocking=True)
        time.sleep(0.1)

        if i == len(poses)-2:
            mh2.call_cli_sync(mh2.finger2srv["right_v2"], RollerGripperV2.Request(position=-0.8))
            time.sleep(0.3)

def placing_sequence(mh2: MotionHelperV2):
    mh2.go_to_q([-1.4939,
        -0.9910,
        2.0959,
        -1.7361,
        1.5754,
        4.0280,
    ], side="right", time=2) # 3
    
    mh2.go_to_q([-1.1826,-0.669801,1.29451,-1.57216,0.941826,4.03527], side="right", time=2) # 2
    mh2.go_to_q([-0.829086,-1.01202,1.98262,-2.05671,0.951,3.88346], side="right", time=1.5) # 1.5
    mh2.go_to_q([-0.731177,-1.07293,1.9299,-1.88504,0.999185,3.78101], side="right", time=0.5) # 0.5

    mh2.call_cli_sync(mh2.finger2srv["right_v2"], RollerGripperV2.Request(position=1.0))
    time.sleep(0.45)
    
    mh2.go_to_q([-0.679236,-1.10291,1.97395,-1.87112,1.0267,3.72891], side="right", time=0.25)
    mh2.go_to_q([-0.740686,-1.06235,2.12228,-2.09265,0.994247,3.7895], side="right", time=0.4)


    mh2.go_to_q([-1.10223,-0.746435,2.00881,-2.5518,0.8478,4.20552], side="right", time=1)

    mh2.go_to_q([-1.4472,-0.5199,2.0481,-2.9698,0.7930,-0.0089], side="right", time=2)

STACK_CHOICE_START = [
    -1.6868,
    -0.3205,
    1.0471,
    -2.9534,
    2.2410,
    2.9875,
]

"""
STACK_CHOICE_START_BOTH 
# Current joint positions for: both [rad]
[
    1.3931,
    -1.8583,
    -2.0507,
    -0.9805,
    -0.7861,
    0.4594,
    -1.6868,
    -0.3205,
    1.0471,
    -2.9534,
    2.2410,
    2.9875,
]
"""

PRE_STACK_START = [
    -2.2036,
    -1.5719,
    2.0296,
    -1.9163,
    2.4260,
    3.8685,
]
"""
PRE_STACK_START_BOTH
# Current joint positions for: both [rad]
[
    1.3931,
    -1.8583,
    -2.0507,
    -0.9805,
    -0.7862,
    0.4593,
    -2.2035,
    -1.5719,
    2.0296,
    -1.9162,
    2.4260,
    3.8685,
]

"""

SAFE_TRANSITION = [
    -1.7321,
    -0.7593,
    1.7561,
    -2.5065,
    1.0454,
    3.0375,
]

"""
SAFE_TRANSITION_BOTH
# Current joint positions for: both [rad]
[
    1.3878,
    -1.8531,
    -2.0427,
    -0.9775,
    -0.7938,
    0.4604,
    -1.7322,
    -0.7593,
    1.7561,
    -2.5064,
    1.0453,
    3.0375,
]


"""

def main(mh2: MotionHelperV2, node: StackDetectorDINO, with_slides=False):
    side = "right"
    input("start?")

    if with_slides: 
        fut = node.cli_start_switch.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(node, fut)

    mh2.go_to_q(
        q=SAFE_TRANSITION,
        time=3,
        side="right"
    )

    if with_slides:
        pre_stack_q = PRE_STACK_START
        mh2.go_to_q(
            q=PRE_STACK_START,
            time=3,
            side="right"
        )
    else:
        mh2.go_to_q(
            q=STACK_CHOICE_START,
            time=2,
            side="right"
        )

        for _ in range(10):
            try:
                pre_stack_pose = start_pose_and_stack_choice(node)
                pre_stack_q = mh2.compute_ik_with_retries(pre_stack_pose, mh2.current_q.copy(), side)
                if not pre_stack_q:
                    print("IK FAILED")
                    continue
                break
            except Exception as e:
                print("Exception during unstacking pose ")

        input("move?")
        mh2.go_to_q(pre_stack_q, 3, side="right")

    print("RUNNING UNSTACKING!")
    time.sleep(0.7) # otherwise we might take the image too early
    unstack(mh2, node)
    print("unstacking done.")

    mh2.go_to_q(
        q=pre_stack_q,
        time=2.5,
        side="right"
    )

    if not with_slides: input("place?")
    placing_sequence(mh2)
   

if __name__ == '__main__':
    import sys

    rclpy.init()
    last_arg = sys.argv[-1]
    with_slides = last_arg == "slides"

    mh2 = MotionHelperV2()
    node = StackDetectorDINO(with_slides=with_slides)

    if with_slides: node.cli_kill_switch.call_async(Trigger.Request())

    try:
        # 3. Run your main logic
        main(mh2=mh2, node=node, with_slides=with_slides)
    except KeyboardInterrupt:
        print("\n[Ctrl+C] detected, shutting down...")
        node.cli_kill_switch.call_async(Trigger.Request())
    except Exception as e:
        print(f"Unstacking Exception:\n{e}")
        node.cli_kill_switch.call_async(Trigger.Request())
    finally:
        # 4. Clean shutdown
        rclpy.shutdown()
    print("bye!")
