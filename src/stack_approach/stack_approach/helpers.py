import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
import rclpy.clock

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