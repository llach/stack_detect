import cv2
import rclpy
import numpy as np

from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, TransformStamped, PointStamped

bridge = CvBridge()
def publish_img(pub, img, frame="camera_color_optical_frame"):
    msg = bridge.cv2_to_compressed_imgmsg(cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB), "jpeg")
    msg.header = Header(
        frame_id=frame,
        stamp=rclpy.clock.Clock().now().to_msg()
    )
    pub.publish(msg)

def pixel_to_point(pixel, depth, K, frame="camera_color_optical_frame"):
    px, py = pixel
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (px- cx) * depth / fx
    y = (py - cy) * depth / fy
    z = depth

    ps = PointStamped()
    ps.header = Header(
        frame_id=frame,
        stamp=rclpy.clock.Clock().now().to_msg()
    )

    ps.point.x = x
    ps.point.y = y
    ps.point.z = z
    
    return ps

def call_cli_sync(node, cli, req):
    fut = cli.call_async(req)
    rclpy.spin_until_future_complete(node, fut)
    return fut.result()
    
def time_msg_to_unix_timestamp(time_msg) -> float:
    # Convert seconds and nanoseconds to a UNIX timestamp with nanosecond precision
    return time_msg.sec + time_msg.nanosec / 1e9

def pose_to_list(p: PoseStamped):
    return [
        time_msg_to_unix_timestamp(p.header.stamp),
        p.header.frame_id,
        [
            p.pose.position.x,
            p.pose.position.y,
            p.pose.position.z,
        ],
        [
            p.pose.orientation.x,
            p.pose.orientation.y,
            p.pose.orientation.z,
            p.pose.orientation.w
        ]
    ]

def transform_to_pose_stamped(tfs: TransformStamped) -> PoseStamped:
    p = PoseStamped()
    p.header = tfs.header
    p.pose.position.x = tfs.transform.translation.x
    p.pose.position.y = tfs.transform.translation.y
    p.pose.position.z = tfs.transform.translation.z
    p.pose.orientation = tfs.transform.rotation
    return p

def empty_pose(frame = ""):
    p = PoseStamped()
    p.header.stamp = rclpy.clock.Clock().now().to_msg()
    p.header.frame_id = frame
    p.pose.position.x = 0.0
    p.pose.position.y = 0.0
    p.pose.position.z = 0.0
    p.pose.orientation.x = 0.0
    p.pose.orientation.y = 0.0
    p.pose.orientation.z = 0.0
    p.pose.orientation.w = 1.0
    return p

def point_to_pose(point):
    pose = PoseStamped()
    pose.header = point.header
    pose.pose.position = point.point
    
    return pose

def grasp_pose_to_wrist(tf_buffer, gp, x_off=0.005, z_off=-0.20) -> PoseStamped:
    p_wrist = tf_buffer.transform(gp, "wrist_3_link")

    pose_wrist = PoseStamped()
    pose_wrist.header = p_wrist.header
    if type(p_wrist) == PoseStamped:
        pose_wrist.pose.position = p_wrist.pose.position
    else:
        pose_wrist.pose.position = p_wrist.point
    pose_wrist.pose.position.x += x_off
    pose_wrist.pose.position.z += z_off
    
    return pose_wrist

def matrix_to_pose_msg(T, frame_id):
        ps = PoseStamped(
            header=Header(
                stamp=rclpy.time.Time().to_msg(),
                frame_id=frame_id
            )
        )
        t = T[:3,3]
        q = R.from_matrix(T[:3,:3]).as_quat()
        ps.pose.position.x = t[0]
        ps.pose.position.y = t[1]
        ps.pose.position.z = t[2]

        ps.pose.orientation.x = q[0]
        ps.pose.orientation.y = q[1]
        ps.pose.orientation.z = q[2]
        ps.pose.orientation.w = q[3]
        
        return ps

def tf_msg_to_matrix(msg):
        t = msg.transform.translation
        t = [t.x, t.y, t.z]

        q = msg.transform.rotation
        q = [q.x, q.y, q.z, q.w]

        T = np.eye(4)
        T[:3,:3] = R.from_quat(q).as_matrix()
        T[:3,3] = t

        return T
    
def pose_to_matrix(msg):
        t = msg.pose.position
        t = [t.x, t.y, t.z]

        q = msg.pose.orientation
        q = [q.x, q.y, q.z, q.w]

        T = np.eye(4)
        T[:3,:3] = R.from_quat(q).as_matrix()
        T[:3,3] = t

        return T
    
def get_trafo(fro, to, tf_buffer):
        return tf_msg_to_matrix(tf_buffer.lookup_transform(fro, to, rclpy.time.Time()))
    
def inv(T):
    Ti = np.eye(4)
    Ti[:3,:3] = T[:3,:3].T
    Ti[:3,3] = -1*T[:3,3]
    return Ti

def finger_matrix_to_wrist_pose(Tf, buffer, frame_id="map"):
    Twf = get_trafo("wrist_3_link", "finger", buffer)
    Tfw = inv(Twf)
    return matrix_to_pose_msg(Tf@Tfw, frame_id)