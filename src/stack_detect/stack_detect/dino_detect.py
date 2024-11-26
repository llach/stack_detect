import os
import cv2
import rclpy
import numpy as np
# import open3d as o3d
from ctypes import * 

from cv_bridge import CvBridge
from threading import Lock

from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

from std_msgs.msg import Header, Int16MultiArray
from sensor_msgs.msg import PointCloud2, PointField, CompressedImage, CameraInfo, Image as ImageMSG

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from PIL import Image

from stack_detect.helpers.dino import load_model, get_grounding_output, plot_boxes_to_image

import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import PointStamped, PoseStamped
from tf2_geometry_msgs import PoseStamped

from tf_transformations import quaternion_matrix


FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]
FIELDS_XYZRGB = FIELDS_XYZ + \
    [PointField(name='rgb', offset=16, datatype=PointField.FLOAT32, count=1)]

# Bit operations
BIT_MOVE_16 = 2**16
BIT_MOVE_8 = 2**8
convert_rgbUint32_to_tuple = lambda rgb_uint32: (
    (rgb_uint32 & 0x00ff0000)>>16, (rgb_uint32 & 0x0000ff00)>>8, (rgb_uint32 & 0x000000ff)
)
convert_rgbFloat_to_tuple = lambda rgb_float: convert_rgbUint32_to_tuple(
    int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value)
)

# Convert the datatype of point cloud from Open3D to ROS PointCloud2 (XYZRGB only)
def convertCloudFromOpen3dToRos(open3d_cloud, frame_id="odom", time=None):
    # Set "header"
    header = Header()
    header.stamp = time
    header.frame_id = frame_id

    # Set "fields" and "cloud_data"
    points=np.asarray(open3d_cloud.points)
    if not open3d_cloud.colors: # XYZ only
        fields=FIELDS_XYZ
        cloud_data=points
    else: # XYZ + RGB
        fields=FIELDS_XYZRGB
        # -- Change rgb color from "three float" to "one 24-byte int"
        # 0x00FFFFFF is white, 0x00000000 is black.
        colors = np.floor(np.asarray(open3d_cloud.colors)*255) # nx3 matrix
        colors = colors[:,0] * BIT_MOVE_16 +colors[:,1] * BIT_MOVE_8 + colors[:,2]  
        cloud_data=np.c_[points, colors]
    
    # create ros_cloud
    return pc2.create_cloud(header, fields, cloud_data)

class StackDetectorDINO(Node):

    def __init__(self, executor):
        self.exe = executor
        super().__init__("StackDetectorDINO")
        self.log = self.get_logger()

        self.bridge = CvBridge()
        self.cb_group = ReentrantCallbackGroup()

        self.depth_sub = self.create_subscription(
            ImageMSG, "/camera/aligned_depth_to_color/image_raw", self.depth_cb, 0, callback_group=MutuallyExclusiveCallbackGroup()
        )
        self.img_sub = self.create_subscription(
            CompressedImage, "/camera/color/image_raw/compressed", self.rgb_cb, 0, callback_group=MutuallyExclusiveCallbackGroup()
        )
        self.info_sub = self.create_subscription(
            CameraInfo, "/camera/color/camera_info", self.info_cb, 0, callback_group=self.cb_group
        )

        self.img_pub = self.create_publisher(CompressedImage, '/segmented_image/compressed', 0, callback_group=self.cb_group)
        self.crop_img_pub = self.create_publisher(CompressedImage, '/cropped_stack/compressed', 0, callback_group=self.cb_group)
        self.pcdpub = self.create_publisher(PointCloud2, '/cropped_cloud', 0, callback_group=self.cb_group)

        self.ppub = self.create_publisher(PointStamped, '/stack_center', 10, callback_group=self.cb_group)
        self.posepub = self.create_publisher(PoseStamped, '/approach_pose', 10, callback_group=self.cb_group)
        self.boxpub = self.create_publisher(Int16MultiArray, '/stack_box', 10, callback_group=self.cb_group)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.depth_lock = Lock()
        self.depth_msg = None
        self.K = None

        ### DINO setup
        self.cpu_only = True
        self.box_threshold = 0.3
        self.text_threshold = 0.25
        self.token_spans = None
        self.text_prompt = "detect all stacks of clothing"
        prefix = f"{os.environ['HOME']}/repos/"
        self.model = load_model(prefix+"GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", prefix+"ckp/groundingdino_swint_ogc.pth", cpu_only=self.cpu_only)

    def publish_img(self, pub, img):
        msg = self.bridge.cv2_to_compressed_imgmsg(cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB), "jpeg")
        msg.header = Header(
            frame_id=self.camera_frame,
            stamp=self.get_clock().now().to_msg()
        )
        pub.publish(msg)

    def transform_point(self, p, frame, target):
        gp = PointStamped()

        gp.header.stamp = self.get_clock().now().to_msg()
        gp.header.frame_id = frame
        gp.point.x = p[0]
        gp.point.y = p[1]
        gp.point.z = p[2]

        return self.tf_buffer.transform(gp, target, timeout=rclpy.duration.Duration(seconds=5))

    def depth_cb(self, msg):
        self.depth_lock.acquire()
        self.depth_msg = msg
        self.depth_lock.release()

    def info_cb(self, msg):
        if self.K is None: 
            self.get_logger().info("got camera calibration")
        else: return
        self.camera_frame = msg.header.frame_id
        self.K = np.reshape(msg.k, (3,3))
        self.H = msg.height
        self.W = msg.width

        d2o = self.tf_buffer.lookup_transform("camera_depth_optical_frame", "camera_color_optical_frame", self.get_clock().now(), timeout=rclpy.duration.Duration(seconds=5))
        self.K_ext = quaternion_matrix([
            d2o.transform.rotation.x,
            d2o.transform.rotation.y,
            d2o.transform.rotation.z,
            d2o.transform.rotation.w,
        ])
        self.K_ext[:3,3] = [
            d2o.transform.translation.x,
            d2o.transform.translation.y,
            d2o.transform.translation.z,   
        ]

    def rgb_cb(self, msg): 
        if self.K is None:
            self.get_logger().warn("no camera calibration yet ...")
            return
        
        if self.depth_msg is None:
            self.get_logger().warn("no depth image yet ...")
            return
        
        ##### Convert
        img = self.bridge.compressed_imgmsg_to_cv2(msg)
        image_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), mode="RGB")
        
        ##### Run DINO
        boxes_px, pred_phrases, _ = get_grounding_output(
            self.model, 
            image_pil, 
            self.text_prompt, 
            self.box_threshold, 
            self.text_threshold, 
            cpu_only=self.cpu_only, 
            token_spans=eval(f"{self.token_spans}")
        )

        if len(boxes_px) == 0:
            self.get_logger().warn("no boxes found!")
            return
        
        image_with_box = plot_boxes_to_image(image_pil.copy(), boxes_px, pred_phrases)[0]

        x0, y0, x1, y1 = boxes_px[0]
        img_crop = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[y0:y1,x0:x1]

        # convert 

        #### Pointcloud
        self.depth_lock.acquire()
        depth = self.bridge.imgmsg_to_cv2(self.depth_msg)
        self.depth_lock.release()

        masked_depth = np.infty*np.ones_like(depth)
        masked_depth[y0:y1,x0:x1] = depth[y0:y1,x0:x1]


        ##### Publish
        self.boxpub.publish(Int16MultiArray(data=[x0, y0, x1, y1]))
        self.publish_img(self.img_pub, image_with_box)
        self.publish_img(self.crop_img_pub, img_crop)
        # self.ppub.publish(stack_center)
        # self.posepub.publish(pose_wrist)
        # self.pcdpub.publish(convertCloudFromOpen3dToRos(pcd, frame_id=self.camera_frame, time=self.get_clock().now().to_msg()))


def main(args=None):
    """Creates subscriber node and spins it"""
    rclpy.init(args=args)

    executor = MultiThreadedExecutor(num_threads=4)
    node = StackDetectorDINO(executor=executor)

    executor.add_node(node)

    try:
        node.get_logger().info('Beginning client, shut down with CTRL-C')
        rclpy.spin(node, executor)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down.\n')

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
