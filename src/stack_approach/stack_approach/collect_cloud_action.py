import os
import gc
import sys
import json
import rclpy
import cv2

import numpy as np

import skvideo.io
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

def get_obj_size(obj):
    marked = {id(obj)}
    obj_q = [obj]
    sz = 0

    while obj_q:
        sz += sum(map(sys.getsizeof, obj_q))

        # Lookup all the object referred to by the object in obj_q.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

        # Filter object that are already marked.
        # Using dict notation will prevent repeated objects.
        new_refr = {o_id: o for o_id, o in all_refr if o_id not in marked and not isinstance(o, type)}

        # The new obj_q will be the ones that were not marked,
        # and we will update marked with their ids so we will
        # not traverse them again.
        obj_q = new_refr.values()
        marked.update(new_refr.keys())

    return sz


class DataCollectionActionServer(Node):
    def __init__(self, store_dir = f"{os.environ['HOME']}/unstack_cloud"):
        super().__init__('data_collection_action_server')
        self.store_dir = store_dir

        self.declare_parameter('sim', False)
        self.declare_parameter('downsample', True)
        self.declare_parameter('imu_topic', "/camera/imu")
        self.declare_parameter('image_topic', "/camera/color/image_raw/compressed")
        self.declare_parameter('depth_topic', "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter('video_dims',  [299, 224])
        self.declare_parameter('crf',  0)

        self.crf = self.get_parameter("crf").get_parameter_value().integer_value
        self.sim = self.get_parameter("sim").get_parameter_value().bool_value
        self.downsample = self.get_parameter("downsample").get_parameter_value().bool_value
        self.video_dims = self.get_parameter("video_dims").get_parameter_value().integer_array_value
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
        if self.collecting_data and self.js_msg is not None:
            # Store joint positions by name
            # self.get_logger().info("JS")
            self.joint_states.append([
                time_msg_to_unix_timestamp(self.js_msg.header.stamp), 
                list(self.js_msg.position)
            ])

    def rgb_cb(self, msg):
        if not self.collecting_data: return
        self.rgb_frames.append(msg2tuple(msg, CompressedImage))

    def depth_cb(self, msg):
        if not self.collecting_data: return
        self.depth_frames.append(msg2tuple(msg, Image))

    def imu_cb(self, msg):
        if not self.collecting_data: return
        self.imu_data.append(msg2tuple(msg, Imu))

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
            n_samples = [
                len(self.joint_states),
                len(self.transforms)
            ]
            if not self.sim:
                n_samples += [
                    len(self.rgb_frames),
                    len(self.depth_frames),
                    len(self.imu_frames)
                ]
            feedback_msg.n_samples

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

            if not self.sim:
                self.get_logger().info(f"IMU frames: {len(self.imu_frames)}")
                with open(f"{sample_dir}/imu.json", "w") as f:
                    f.write(json.dumps(self.imu_frames))

                self.get_logger().info(f"RGB frames: { len(self.rgb_frames)} | {get_obj_size(self.rgb_frames)*10**-6:.2f}MB (raw)")
                writer = skvideo.io.FFmpegWriter(f"{sample_dir}/rgb.mp4", outputdict={
                    '-vcodec': 'libx264',  #use the h.264 codec
                    '-crf': '0',           #set the constant rate factor to 0, which is lossless
                    '-preset':'veryslow'   #the slower the better compression, in princple, try 
                                        #other options see https://trac.ffmpeg.org/wiki/Encode/H.264
                }) 
                rgb_stamps = []
                for t, frame in self.rgb_frames:
                    rgb_stamps.append(t)
                    if self.downsample: frame = cv2.resize(frame, self.video_dims, interpolation = cv2.INTER_AREA)
                    writer.writeFrame(frame)
                writer.close()

                with open(f"{sample_dir}/rgb_stamps.json", "w") as f:
                    f.write(json.dumps(rgb_stamps))

                self.get_logger().info(f"Depth frames: {len(self.rgb_frames)}| {get_obj_size(self.depth_frames)*10**-6:.2f}MB (raw)")
                writer = skvideo.io.FFmpegWriter(f"{sample_dir}/depth.mp4", outputdict={
                    '-vcodec': 'libx264',  #use the h.264 codec
                    '-crf': '0',           #set the constant rate factor to 0, which is lossless
                    '-preset':'veryslow'   #the slower the better compression, in princple, try 
                                        #other options see https://trac.ffmpeg.org/wiki/Encode/H.264
                }) 
                depth_stamps = []
                for t, dframe in self.depth_frames:
                    depth_stamps.append(t)

                    dframe = (np.clip(dframe, 0, 2000)/2000*255).astype(np.uint8)
                    if self.downsample: dframe = cv2.resize(dframe, self.video_dims, interpolation = cv2.INTER_AREA)
                    
                    df = np.zeros(dframe.shape[:2]+(3,), dtype=np.uint8)
                    df[:,:,0] = dframe

                    writer.writeFrame(dframe)
                writer.close()

                with open(f"{sample_dir}/depth_stamps.json", "w") as f:
                    f.write(json.dumps(depth_stamps))

            self.clear_buffers()
            self.get_logger().info("done saving.")
            
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
