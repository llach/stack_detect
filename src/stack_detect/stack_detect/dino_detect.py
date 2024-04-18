import os
import cv2
import rclpy
import numpy as np
import open3d as o3d
from ctypes import * 

from cv_bridge import CvBridge
from threading import Lock

from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

from rclpy.qos import QoSProfile
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField, CompressedImage, CameraInfo, Image as ImageMSG

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

import torch
from PIL import Image, ImageDraw, ImageFont

from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.vl_utils import create_positive_map_from_span
import groundingdino.datasets.transforms as T

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

def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    
    text_prompt = caption
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
    else:
        # given-phrase mode
        positive_maps = create_positive_map_from_span(
            model.tokenizer(text_prompt),
            token_span=token_spans
        ).to(image.device) # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            # get phrase
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > box_threshold
            # filt box
            all_boxes.append(boxes[filt_mask])
            # filt logits
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
            else:
                all_phrases.extend([phrase for _ in range(len(filt_mask))])
        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        pred_phrases = all_phrases


    return boxes_filt, pred_phrases

def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple((255, 0, 255))
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


class StackDetectorDINO(Node):

    def __init__(self, executor):
        self.exe = executor
        super().__init__("StackDetectorDINO")
        self.log = self.get_logger()

        self.bridge = CvBridge()

        self.cb_group = ReentrantCallbackGroup()

        # rqt image view doesn't work with different qos profiles, because why would it? why would a vital part of ROS infatructure still work seamlessly in ROS2? it's fuck usability now.
        # qos_profile = QoSProfile(depth=1)
        # qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        # qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT
        # qos_profile.durability = QoSDurabilityPolicy.VOLATILE

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

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.depth_lock = Lock()
        self.depth_msg = None
        self.K = None

        ### DINO setup
        self.cpu_only = False
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
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w

        boxes_filt, pred_phrases = get_grounding_output(
            self.model, 
            image, 
            self.text_prompt, 
            self.box_threshold, 
            self.text_threshold, 
            cpu_only=self.cpu_only, 
            token_spans=eval(f"{self.token_spans}")
        )

        if len(boxes_filt) == 0:
            self.get_logger.warn("no boxes found!")
            return

        size = image_pil.size
        W, H = size
        pred_dict = {
            "boxes": boxes_filt,
            "size": [size[1], size[0]],  # H,W
            "labels": pred_phrases,
        }
        image_with_box = plot_boxes_to_image(image_pil.copy(), pred_dict)[0]

        box = boxes_filt[0] * torch.Tensor([W, H, W, H])
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]

        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        
        img_crop = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[y0:y1,x0:x1]

        #### Pointcloud
        self.depth_lock.acquire()
        depth = self.bridge.imgmsg_to_cv2(self.depth_msg)
        self.depth_lock.release()

        masked_depth = np.infty*np.ones_like(depth)
        masked_depth[y0:y1,x0:x1] = depth[y0:y1,x0:x1]

        rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))),
            o3d.geometry.Image(masked_depth.astype(np.uint16)),
            convert_rgb_to_intensity=False
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            image=rgbd_img,
            intrinsic=o3d.camera.PinholeCameraIntrinsic(
                width=self.W,
                height=self.H,
                intrinsic_matrix=self.K
            ),
            extrinsic=self.K_ext.T
        )

        # print(pcd.colors[0])
        # o3d.visualization.draw_geometries(
        #     [pcd], 
        #     point_show_normal=False,
        # )

        stack_center = np.mean(pcd.points, axis=0)
        stack_center = self.transform_point(stack_center, self.camera_frame, "map")

        pose_wrist = PoseStamped()
        pose_wrist.header.stamp = self.get_clock().now().to_msg()
        pose_wrist.header.frame_id = "map"
        pose_wrist.pose.position.x = stack_center.point.x - 0.15
        pose_wrist.pose.position.y = stack_center.point.y - 0.50
        pose_wrist.pose.position.z = stack_center.point.z
        
        tgt_quat = [0.519, 0.508, 0.491, -0.482]
        pose_wrist.pose.orientation.x = tgt_quat[0]
        pose_wrist.pose.orientation.y = tgt_quat[1]
        pose_wrist.pose.orientation.z = tgt_quat[2]
        pose_wrist.pose.orientation.w = tgt_quat[3]

        ##### Publish
        self.publish_img(self.img_pub, image_with_box)
        self.publish_img(self.crop_img_pub, img_crop)
        self.ppub.publish(stack_center)
        self.posepub.publish(pose_wrist)
        self.pcdpub.publish(convertCloudFromOpen3dToRos(pcd, frame_id=self.camera_frame, time=self.get_clock().now().to_msg()))

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
