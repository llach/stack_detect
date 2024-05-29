import os
import cv2
import time
import rclpy
import numpy as np
import open3d as o3d
from ctypes import * 

from cv_bridge import CvBridge
from threading import Lock

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
cm = list(mcolors.TABLEAU_COLORS.values())

from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

from std_msgs.msg import Header, Int16MultiArray
from sensor_msgs.msg import PointCloud2, PointField, CompressedImage, CameraInfo, Image as ImageMSG

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from collections import OrderedDict

import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import PointStamped, PoseStamped
from tf2_geometry_msgs import PoseStamped

from tf_transformations import quaternion_matrix

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

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

tab = [
    [78, 121, 167],
    [242, 142, 43],
    [225, 87, 89],
    [118, 183, 178],
    [89, 161, 79],
    [237, 201, 72],
    [176, 122, 161],
    [255, 157, 167],
    [156, 117, 95],
    [186, 176, 172]
]

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

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.random.random(3)*255
        img[m] = color_mask
    return img

def mask_image(anns):
    if len(anns) == 0:
        return

    col = [
        [255,0,0],
        [0,255,0],
        [0,0,255]
    ]
    img = 255*np.ones((anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1], 3))
    for i, ann in enumerate(anns):
        m = ann['segmentation']
        color_mask = col[i%len(col)]
        img[m] = color_mask
        if i==2:break
    return img

def mask_center(mask):
    sumx, sumy = 0,0
    nele = np.sum(mask)
    for i, col in enumerate(mask):
        for j, el in enumerate(col):
            if el:
                sumx += i
                sumy += j
    return [int(sumy/nele), int(sumx/nele)]


class StackDetectorSAM(Node):

    def __init__(self, executor):
        self.exe = executor
        super().__init__("StackDetectorSAM")
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
        self.box_sub = self.create_subscription(
            Int16MultiArray, '/stack_box', self.box_cb, 0, callback_group=MutuallyExclusiveCallbackGroup()    
        )

        self.img_pub = self.create_publisher(CompressedImage, '/sam_image/compressed', 0, callback_group=self.cb_group)
        self.pcdpub = self.create_publisher(PointCloud2, '/sam_cloud', 0, callback_group=self.cb_group)

        self.ppub = self.create_publisher(PointStamped, '/grasp_point', 10, callback_group=self.cb_group)
        self.posepub = self.create_publisher(PoseStamped, '/grasp_pose', 10, callback_group=self.cb_group)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.depth_lock = Lock()
        self.depth_msg = None
        self.img_lock = Lock()
        self.img_msg = None
        self.K = None

        ### SAM setup
        sam_checkpoint, model_type = f"{os.environ['HOME']}/repos/ckp/sam_vit_l_0b3195.pth", "vit_l"

        self.get_logger().info("loading model")
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device="cuda")

        self.get_logger().info("creating mask generator")
        self.mask_generator = SamAutomaticMaskGenerator(
            self.sam,
            # points_per_side=64,
            # points_per_batch=32,
        )
        self.get_logger().info("setup done!")

    def create_pcd(self, rgb, depth):
        rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb.astype(np.uint8)),
            o3d.geometry.Image(depth.astype(np.uint16)),
            convert_rgb_to_intensity=False
        )

        return o3d.geometry.PointCloud.create_from_rgbd_image(
            image=rgbd_img,
            intrinsic=o3d.camera.PinholeCameraIntrinsic(
                width=self.W,
                height=self.H,
                intrinsic_matrix=self.K
            ),
            extrinsic=self.K_ext.T
        )

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
        
        self.img_lock.acquire()
        self.img_msg = msg
        self.img_lock.release()

        # the remainder is just for debugging without DINO (faster)
        return

        self.depth_lock.acquire()
        depth = self.bridge.imgmsg_to_cv2(self.depth_msg)
        self.depth_lock.release()
        depth_crop = depth[:,200:400]

        img = self.bridge.compressed_imgmsg_to_cv2(self.img_msg)[:,200:400,:]
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        start = time.time()
        masks = self.mask_generator.generate(img)
        self.get_logger().info(f"took {time.time()-start}; found {len(masks)} masks")

        if len(masks) == 0:
            self.get_logger().warn("no masks found!")
            return
        
        dists = np.array([np.sum(depth_crop[m["segmentation"]])/m["area"] for m in masks])
        mean_dist = 1.5*np.median(dists)
        filtered_masks = [m for m in masks if np.sum(depth_crop[m["segmentation"]])/m["area"] < mean_dist and m["area"] > 5000]
        centers = [mask_center(sa["segmentation"]) for sa in filtered_masks]

        print("#filtered", len(filtered_masks))
        if len(filtered_masks)==0:
            print("no masks after filtering!")
            return
        # for d, fm in zip(dists, filtered_masks): print(fm["area"], d)

        sorted_anns = sorted(filtered_masks, key=(lambda x: mask_center(x["segmentation"])[0]), reverse=True)
        for c in centers: print(c)
        mask_img = mask_image([sorted_anns[0]]).astype(np.uint8)

        clust = sorted_anns[0]["segmentation"]
        max_idxs = clust.shape[1] - np.argmax(clust[:, ::-1], axis=1) - 1
        for i, m in enumerate(max_idxs): mask_img[i,m] = [255, 0, 255]

        msk_w = 0.6
        overlay = np.clip((1-msk_w)*img + msk_w*mask_img, 0, 255).astype(np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, sa in enumerate(sorted_anns):
            p = mask_center(sa["segmentation"])
            print(p)
            cv2.putText(overlay,f'{i}', p, font, 1,(255,255,255),2,cv2.LINE_AA)
        
        self.publish_img(self.img_pub, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    def box_cb(self, msg): 
        self.log.info("got new box!")

        if self.K is None:
            self.get_logger().warn("no camera calibration yet ...")
            return
        
        if self.depth_msg is None:
            self.get_logger().warn("no depth image yet ...")
            return
        
        if self.img_msg is None:
            self.get_logger().warn("no rgb image yet ...")
            return
        
        ##### Convert RGB image
        self.img_lock.acquire()
        img = self.bridge.compressed_imgmsg_to_cv2(self.img_msg)
        self.img_lock.release()
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        ##### Run SAM
        x0, y0, x1, y1 = msg.data
        img_crop = img[y0:y1,x0:x1]

        start = time.time()
        masks = self.mask_generator.generate(img_crop)
        self.get_logger().info(f"took {time.time()-start}; found {len(masks)} masks")

        if len(masks) == 0:
            self.get_logger().warn("no masks found!")
            return
        
        # filter annotations by distance and size
        self.depth_lock.acquire()
        depth = self.bridge.imgmsg_to_cv2(self.depth_msg)
        self.depth_lock.release()
        depth_crop = depth[y0:y1,x0:x1]
        h = y1-y0
        
        dists = np.array([np.sum(depth_crop[m["segmentation"]])/m["area"] for m in masks])
        mean_dist = 1.5*np.median(dists)
        centers = [mask_center(sa["segmentation"]) for sa in masks]
        filtered_masks = [m for c, m in zip(centers, masks) if np.sum(depth_crop[m["segmentation"]])/m["area"] < mean_dist and m["area"] > 5000 and c[1]>0.25*h and c[1]<0.75*h]

        if len(filtered_masks) == 0:
            self.get_logger().warn("no masks after filtering!")
            return

        sorted_masks = sorted(filtered_masks, key=(lambda x: mask_center(x["segmentation"])[0]), reverse=True)
        centers = [mask_center(sa["segmentation"]) for sa in sorted_masks]

        for c in centers: print(c)
        mask_img = mask_image([sorted_masks[0]]).astype(np.uint8)

        # get topmost cluster
        clust = sorted_masks[0]["segmentation"]
        # select bottom pixels
        max_idxs = np.argmax(clust, axis=1) - 1
        # paint them pink
        for i, m in enumerate(max_idxs): mask_img[i,m] = [255, 0, 255]

        # superimpose masks on cropped image
        msk_w = 0.6
        overlay = np.clip((1-msk_w)*img_crop + msk_w*mask_img, 0, 255).astype(np.uint8)

        # annotate cluster indices in their centers
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, c in enumerate(centers):
            cv2.putText(overlay,f'{i}', c, font, 1,(255,255,255),2,cv2.LINE_AA)
        
        # insert overlayed image crop onto real image
        img[y0:y1,x0:x1] = overlay

        #### Pointcloud
        limg = 255*np.ones_like(img)
        for i, m in enumerate(max_idxs): limg[i,m] = [255, 0, 255]

        ldepth = np.infty*np.ones_like(depth)
        for i, m in enumerate(max_idxs): ldepth[y0:y1,x0:x1][i,m] = depth[y0:y1,x0:x1][i,m]

        line_pcd = self.create_pcd(limg, ldepth)
        line_pcd.paint_uniform_color([0,0,1])


        # masked_depth = np.infty*np.ones_like(depth)
        # masked_depth[y0:y1,x0:x1] = depth[y0:y1,x0:x1]
 

        # print(len(pcd.points), len(np.where(np.array(pcd.colors)==[0,0,1])[0]))
        # line_pcd = pcd.select_by_index(np.where(np.array(pcd.colors)==[0,0,1])[0])
        # print(len(line_pcd.points))
       
        line_center = np.median(line_pcd.points, axis=0)

        center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        center_sphere.translate(line_center)

        line_center = self.transform_point(line_center, self.camera_frame, "map")
        line_center.point.x

        pose_wrist = PoseStamped()
        pose_wrist.header.stamp = self.get_clock().now().to_msg()
        pose_wrist.header.frame_id = "map"
        pose_wrist.pose.position.x = line_center.point.x
        pose_wrist.pose.position.y = line_center.point.y - 0.18
        pose_wrist.pose.position.z = line_center.point.z - 0.025
        
        tgt_quat = [0.519, 0.508, 0.491, -0.482]
        pose_wrist.pose.orientation.x = tgt_quat[0]
        pose_wrist.pose.orientation.y = tgt_quat[1]
        pose_wrist.pose.orientation.z = tgt_quat[2]
        pose_wrist.pose.orientation.w = tgt_quat[3]

        ##### Publish
        self.publish_img(self.img_pub, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # self.publish_img(self.crop_img_pub, img_crop)
        self.ppub.publish(line_center)
        self.posepub.publish(pose_wrist)
        self.pcdpub.publish(convertCloudFromOpen3dToRos(line_pcd, frame_id=self.camera_frame, time=self.get_clock().now().to_msg()))

        return

        # image = cv2.imdecode(msg.data)

        image_pil = Image.fromarray(cv_img).convert("RGB")  # load image
        image_pil.show()

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w

        image_pil.show()
  

def main(args=None):
    """Creates subscriber node and spins it"""
    rclpy.init(args=args)

    executor = MultiThreadedExecutor(num_threads=4)
    node = StackDetectorSAM(executor=executor)

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
