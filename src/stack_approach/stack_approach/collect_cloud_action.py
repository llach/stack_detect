import os
import json
import rclpy
import cv2

import numpy as np

from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from tf2_ros import TransformListener, Buffer
from stack_msgs.action import RecordCloud  # Import the renamed action
import threading
from datetime import datetime
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import CompressedImage, Image, Imu


from stack_msgs.srv import StoreData
from stack_approach.helpers import transform_to_pose_stamped, pose_to_list, time_msg_to_unix_timestamp

from cv_bridge import CvBridge

bridge = CvBridge()
def msg2tuple(msg, msgtype):

    ts = msg.header.stamp
    t = time_msg_to_unix_timestamp(msg.header.stamp)
    
    if msgtype == Imu: # => (t, frame_id, linear_acceleration, angular_velocity)
        la = msg.linear_acceleration
        av = msg.angular_velocity
        return (t, msg.header.frame_id, [la.x, la.y, la.z], [av.x, av.y, av.z])
    if msgtype == CompressedImage: # => (t, img)
        img = cv2.cvtColor(bridge.compressed_imgmsg_to_cv2(msg), cv2.COLOR_RGB2BGR).astype(np.uint8)
        return (t, img)
    if msgtype == Image: # => (t, img)
        img = bridge.imgmsg_to_cv2(msg).astype(np.uint16)
        return (t, img)

    assert False, f"unknown message type \"{msgtype}\""

class DataCollectionActionServer(Node):
    def __init__(self, store_dir = f"{os.environ['HOME']}/unstack_cloud"):
        super().__init__('data_collection_action_server')
        self.store_dir = store_dir

        self.declare_parameter('sim', False)
        self.declare_parameter('imu_topic', "/imu")
        self.declare_parameter('image_topic', "/image")
        self.declare_parameter('depth_topic', "/depth")

        self.sim = self.get_parameter("sim").get_parameter_value().bool_value
        self.imu_topic = self.get_parameter("imu_topic").get_parameter_value().string_value
        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value

        self.cbg = ReentrantCallbackGroup()

        # Action server setup
        self._action_server = ActionServer(
            self,
            RecordCloud,  # Use the renamed action
            'collect_cloud_data',  # Action name
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            handle_accepted_callback=self.handle_accepted_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.cbg
        )

        # Transform listener setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Storage for joint states and transforms
        self.joint_names = None
        self.js_msg = None
        self.joint_states = []
        self.transforms = []
        self.rgb_frames = []
        self.depth_frames = []
        self.imu_frames = []

        # Subscription to joint states
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10,
            callback_group=self.cbg
        )
        self.timer = self.create_timer(1.0 / 30.0, self.js_timer)

        if not self.sim:
            self.rgb_sub = self.create_subscription(
                CompressedImage,
                self.image_topic,
                self.rgb_cb,
                0
            )
            self.depth_sub = self.create_subscription(
                Image,
                self.depth_topic,
                self.depth_cb,
                0
            )
            self.imu_sub = self.create_subscription(
                Imu,
                self.imu_topic,
                self.imu_cb,
                0
            )

        self.srv = self.create_service(StoreData, 'store_cloud_data', self.srv_callback)

        # Control flags for data collection
        self.collecting_data = False
        print("accepting goals!")

    def goal_callback(self, goal_request):
        # Accept all incoming goals
        self.get_logger().info("Received goal request for data collection.")
        self.clear_buffers()
        return GoalResponse.ACCEPT

    def handle_accepted_callback(self, goal_handle):
        # Execute the goal in a new coroutine to allow for concurrency
        goal_handle.execute()

    def cancel_callback(self, goal_handle):
        self.get_logger().info("Cancelling data collection...")

        self.collecting_data = False
        return CancelResponse.ACCEPT

    def joint_state_callback(self, msg):
        if self.collecting_data:
            if self.joint_names is None: self.joint_names = msg.name
            self.js_msg = msg

    def js_timer(self):
        if self.collecting_data:
            # Store joint positions by name
            # self.get_logger().info("JS")
            self.joint_states.append([
                time_msg_to_unix_timestamp(self.js_msg.header.stamp), 
                list(self.js_msg.position)
            ])

    def rgb_cb(self, msg):
        if not self.collecting_data: return
        self.rgb_frames.append(self.msg2tuple(msg, CompressedImage))

    def depth_cb(self, msg):
        if not self.collecting_data: return
        self.depth_frames.append(self.msg2tuple(msg, Image))

    def imu_cb(self, msg):
        if not self.collecting_data: return
        self.imu_data.append(self.msg2tuple(msg, Imu))

    def get_transform_data(self):
        try:
            transform = self.tf_buffer.lookup_transform('map', 'wrist_3_link', rclpy.time.Time())
            timestamp = int(datetime.now().timestamp())

            return pose_to_list(transform_to_pose_stamped(transform))
        except Exception as e:
            self.get_logger().warn(f"Transform not found: {e}")
            return None

    def execute_callback(self, goal_handle):
        self.get_logger().info('Starting data collection action...')

        # Initialize data collection
        self.collecting_data = True

        # Loop until the goal is canceled
        r = self.create_rate(30)
        while rclpy.ok() and self.collecting_data:
            transform_data = self.get_transform_data()
            if transform_data:
                self.transforms.append(transform_data)

            # Send feedback with the count of collected samples
            feedback_msg = RecordCloud.Feedback()
            feedback_msg.n_samples = [
                len(self.joint_states),
                len(self.transforms)
            ]
            if not self.sim:
                feedback_msg.n_samples += [
                    len(self.rgb_frames),
                    len(self.depth_frames),
                    len(self.imu_frames)
                ]

            goal_handle.publish_feedback(feedback_msg)

            # self.get_logger().info("TF")
            # Allow time for new joint state messages
            r.sleep()

        # Stop data collection and complete the action
        self.collecting_data = False
        self.get_logger().info('Data collection canceled.')

        result = RecordCloud.Result()
        result.success = True
        goal_handle.succeed()
        return result

    def clear_buffers(self):
        """Clear the data buffers for joint states and transforms."""
        self.joint_names = None
        self.js_msg = None
        self.joint_states = []
        self.transforms = []
        self.rgb_frames = []
        self.depth_frames = []
        self.imu_frames = []
        self.get_logger().info("Data buffers cleared.")

    def srv_callback(self, req, res):
        self.get_logger().info(f"saving Nsamples:\njs: {len(self.joint_states)} | tf: {len(self.transforms)}")

        sample_dir = req.dir

        try:
            with open(sample_dir+"joint_states.json", "w") as f:
                json.dump({
                    "names": self.joint_names,
                    "joint_states": self.joint_states
                }, f)

            with open(sample_dir+"gripper_poses.json", "w") as f:
                json.dump(self.transforms, f)

            self.clear_buffers()
            print("done saving.")
            
        except Exception as e:
            self.get_logger().fatal("Couldn't store data:\n"+e)
            res.success = False
            return res
        res.success = True
        return res
            

def main(args=None):
    rclpy.init(args=args)

    node = DataCollectionActionServer()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
