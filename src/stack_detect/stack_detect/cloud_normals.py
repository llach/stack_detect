"""Subscriber module"""
import open3d as o3d
import rclpy
import numpy as np
import time
import struct
import ctypes
from ctypes import *
from datetime import datetime

import threading 
from rcl_interfaces.srv import GetParameters
from control_msgs.action import FollowJointTrajectory
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import TransformStamped, PointStamped, PoseStamped
from tf2_geometry_msgs import PoseStamped
from sensor_msgs.msg import JointState
from moveit_msgs.msg import DisplayRobotState
import ros2_numpy as rnp

from tf2_ros import TransformBroadcaster
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points_numpy, read_points, create_cloud
import tf_transformations as tf

from matplotlib import colormaps as cm

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from moveit_msgs.srv import GetPositionIK
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
from rclpy.qos import QoSProfile

VIEW_PARAMS = { 
    "front" : [ 0.12346865912958778, -0.20876634334886676, 0.97014024970489954 ],
    "lookat" : [ 0.012304816534506125, -0.0072898239635080433, 0.21474424582004883 ],
    "up" : [ 0.9914498309557469, 0.06754698721022645, 0.11164513969108841 ],
    "zoom" : 0.35999999999999965
}

convert_rgbUint32_to_tuple = lambda rgb_uint32: (
    (rgb_uint32 & 0x00ff0000)>>16, (rgb_uint32 & 0x0000ff00)>>8, (rgb_uint32 & 0x000000ff)
)
convert_rgbFloat_to_tuple = lambda rgb_float: convert_rgbUint32_to_tuple(
    int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value)
)

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
# The data structure of each point in ros PointCloud2: 16 bits = x + y + z + rgb
FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]
FIELDS_XYZRGB = FIELDS_XYZ + \
    [PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1)]

BIT_MOVE_16 = 2**16
BIT_MOVE_8 = 2**8

# Convert the datatype of point cloud from Open3D to ROS PointCloud2 (XYZRGB only)
def convertCloudFromOpen3dToRos(open3d_cloud, frame_id):
    # Set "header"
    header = Header()
    header.stamp = rclpy.time.Time().to_msg()
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
        # colors = colors.astype(np.float32)
        cloud_data=np.c_[points, colors]
    
    # create ros_cloud
    return create_cloud(header, fields, cloud_data)


def convertCloudFromRosToOpen3d(ros_cloud):
    # this method needs a lot of cleaning.

    # Get cloud data from ros_cloud
    field_names=[field.name for field in ros_cloud.fields]
    cloud_data = list(read_points(ros_cloud, skip_nans=True, field_names = field_names))

    # Check empty
    open3d_cloud = o3d.geometry.PointCloud()
    if len(cloud_data)==0:
        print("Converting an empty cloud")
        return None

    # Set open3d_cloud
    if "rgb" in field_names:
        IDX_RGB_IN_FIELD=3 # x, y, z, rgb
        
        # Get xyz
        xyz = [(x,y,z) for x,y,z,rgb in cloud_data ] # (why cannot put this line below rgb?)

        # Get rgb
        # Check whether int or float
        if type(cloud_data[0][IDX_RGB_IN_FIELD])==float: # if float (from pcl::toROSMsg)
            rgb = [convert_rgbFloat_to_tuple(rgb) for x,y,z,rgb in cloud_data ]
        else:
            rgb = [convert_rgbFloat_to_tuple(rgb) for x,y,z,rgb in cloud_data ]

        # combine
        open3d_cloud.points = o3d.utility.Vector3dVector(np.array(xyz))
        open3d_cloud.colors = o3d.utility.Vector3dVector(np.array(rgb)/255.0)
    else:
        xyz = [(x,y,z) for x,y,z in cloud_data ] # get xyz
        open3d_cloud.points = o3d.utility.Vector3d(np.array(xyz))

    # return
    return open3d_cloud

def sort_list_by_indices(li, idxs):
    li = [li[i] for i in idxs]
    return li

def random_growing_clusters(pcd, cond, k=10, min_size=500):
    C = [] #cluster
    idxs = list(cond.nonzero()[0]) # indices of all points that match the condition
    tree = o3d.geometry.KDTreeFlann(pcd)
    labels = np.zeros(len(pcd.points), dtype=np.int8)

    def add_neighbors(p, Ci, idxs):
        # recursively add all neighbors that are fulfill the condition and are not assigned to a cluster to C_i
        Ci.append(p)
        idxs.remove(p)

        [_, Nidxs, _] = tree.search_knn_vector_3d(pcd.points[p], k)
        for Nj in Nidxs: 
            if Nj in idxs and not Nj in Ci: idxs, Ci = add_neighbors(Nj, Ci, idxs)
        return idxs, Ci

    # build clusters
    while len(idxs)>0:
        Ci = []
        
        prand = idxs[np.random.choice(len(idxs))]
        idxs, Ci = add_neighbors(prand, Ci, idxs)

        C.append(Ci)

    # filter cluster based on size
    C = [Ci for Ci in C if len(Ci)>=min_size]

    # print(f"found {len(C)} cluster")
    for i, Ci in enumerate(C):
        for pi in Ci: labels[pi] = i+1
        # print(f"\t{i}: {len(Ci)}")

    return C, labels

def cloud_var(pcd):
    points = np.array(pcd.points)
    return np.abs(np.min(points, axis=0) - np.max(points, axis=0))


def remove_points_on_plane_and_below(pcd, log, cam_direction=2):
    geoms = []

    # get inliers and outliers, paint for debugging
    _, inliers = pcd.segment_plane(distance_threshold=0.01,
                                            ransac_n=3,
                                            num_iterations=1000)
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    points = np.array(pcd.points)
    geoms += [inlier_cloud, outlier_cloud]

    got_ground_plane = cloud_var(inlier_cloud)[cam_direction] >  cloud_var(outlier_cloud)[cam_direction]

    if got_ground_plane:
        log.debug("got ground plane")
        # find the highest point in the plane (on whichever axis is up) and remove all points lower than that
        highest_plane_point = np.max(np.array(inlier_cloud.points), axis=0)
        higher_indices = np.where(points[:,0]>highest_plane_point[0])[0]
    else:
        log.debug("got vertical (stack) plane")
        closest_plane_point = np.min(np.array(inlier_cloud.points)[:,cam_direction])
        higher_indices = np.where(points[:,cam_direction]>closest_plane_point)[0]

    pcd = pcd.select_by_index(higher_indices)
    geoms.append(pcd)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
    geoms.append(mesh_frame)

    # o3d.visualization.draw_geoms(
    #     geoms, 
    #     point_show_normal=False,
    #     # **VIEW_PARAMS
    # )

    return pcd


class StackDetector3D(Node):
    """Subscriber node"""

    ROS_TIP = [
        "tip_frame",
        [0.012, 0.0, 0.181],
        [-0.5, -0.5, 0.5, 0.5],
    ]

    TRAJ_CTRL = "scaled_joint_trajectory_controller"

    def __init__(self, executor):
        self.exe = executor
        super().__init__("StackDetector3D")
        self.log = self.get_logger()

        self.cb_group = ReentrantCallbackGroup()

        qos_profile = QoSProfile(depth=1)
        qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT
        qos_profile.durability = QoSDurabilityPolicy.VOLATILE

        self.subscription = self.create_subscription(
            PointCloud2, "/camera/depth/color/points", self.point_cb, qos_profile, callback_group=MutuallyExclusiveCallbackGroup()
        )

        self.tf_broadcaster = TransformBroadcaster(self)
        self.timer = self.create_timer(0.1, self.broadcast_timer_callback)

        self.ppub = self.create_publisher(PointStamped, '/grasp_point', 10, callback_group=self.cb_group)
        self.posepub = self.create_publisher(PoseStamped, '/grasp_pose', 10, callback_group=self.cb_group)
        self.pcdpub = self.create_publisher(PointCloud2, '/segmented_cloud', 0, callback_group=self.cb_group)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def js_callback(self, msg): 
        self.current_q = {jname: q for jname, q in zip(msg.name, msg.position)}

    def broadcast_timer_callback(self):
       t = TransformStamped()

       t.header.stamp = self.get_clock().now().to_msg()
       t.header.frame_id = 'hand_e_link'
       t.child_frame_id = 'tip_frame'
       t.transform.translation.x = 0.012
       t.transform.translation.y = 0.0
       t.transform.translation.z = 0.181
       t.transform.rotation.x = -0.5
       t.transform.rotation.y = -0.5
       t.transform.rotation.z = 0.5
       t.transform.rotation.w = 0.5

       self.tf_broadcaster.sendTransform(t)

    def create_traj(self, qfinal, time):
        traj_goal = FollowJointTrajectory.Goal()
        traj_goal.trajectory = JointTrajectory(
            joint_names=list(qfinal.keys()),
            points=[
                JointTrajectoryPoint(
                    positions=list(qfinal.values()),
                    time_from_start=rclpy.duration.Duration(seconds=time).to_msg()
                )
            ]
        )
        return traj_goal

    def point_cb(self, msg):
        # axes indices AFTER transforming for normal vector estimation
        HEIGHT_AXS = 0
        WIDTH_AXS = 1
        DEPTH_AXS = 2
        t = rclpy.time.Time.from_msg(msg.header.stamp)
        dt = datetime.fromtimestamp(t.nanoseconds // 1000000000)
        self.get_logger().debug(f"got data at {dt}")
        start = time.time()
        geoms = []

        try:
            pcd = convertCloudFromRosToOpen3d(msg)
            rgb = np.array(pcd.colors)
        except:
            self.get_logger().warn("np pcd")
            return

        if len(pcd.points)<500:
            self.get_logger().warn("too few points")
            return
        
        # step I: crop cloud
        bb_min, bb_max = [-0.2,-0.2,0], [0.2, 0.2, 0.5]
        aabb = o3d.geometry.AxisAlignedBoundingBox(bb_min, bb_max)
        aabb.color = [1,0,0]

        pcd = pcd.rotate(tf.rotation_matrix(np.pi, [0,0,1])[:3,:3], center=[0,0,0])
        pcd = pcd.crop(aabb)

        # # step II: some pcds have a lot of noise, so we remove outliers based on density. not doing so can result in suboptimal plane estimates
        pcd, _ = pcd.remove_radius_outlier(nb_points=250, radius=0.02)

        # step III: remove supporting plane and points below
        try:
            pcd = remove_points_on_plane_and_below(pcd, log=self.get_logger()) 
        except Exception as e:
            self.get_logger().warn(f"failed to detect plane\n{e}")
            return
        pcd, _ = pcd.remove_radius_outlier(nb_points=250, radius=0.02) # cutting can leave small clusters behind, so we filter them again

        # table
        # Rz = tf.rotation_matrix(np.pi/2, [0,0,1])[:3,:3]
        # Ry = tf.rotation_matrix(np.pi, [0,1,0])[:3,:3]
        # R = Ry@Rz

        # shelf
        R = tf.rotation_matrix(np.pi, [1,0,0])[:3,:3]

        pcd = pcd.rotate(R, center=[0,0,0])
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=0.005))

        nls = np.array(pcd.normals)[:]

        # project normals to xz-plane
        angs = np.clip(np.arctan2(nls[:,0], nls[:,2]), -np.pi/2, np.pi/2)+(np.pi/2)
        angs /= np.pi

        if np.sum(np.all([angs<0.45, angs>0.05], axis=0)) == 0:
            self.get_logger().warn("no normals found to match criteria")
            return
        
        # pcd = pcd.rotate(R.T, center=[0,0,0])

        # color pcd based on angles
        sms = cm["magma"]
        colors = np.array(sms(angs))

        colors = np.array(pcd.colors)
        colors_angles = sms(angs)[:,:3]
        for i, ang in enumerate(angs):
            # colors[i] = colors_angles[i]
            if ang < 0.45 and ang > 0.05: colors[i] = colors_angles[i]

        points = np.array(pcd.points)
        stack_center = np.mean(points, axis=0)
       
        try:
            C, labels = random_growing_clusters(pcd, np.all([angs<0.45, angs>0.05], axis=0), k=20, min_size=5)
        except:
            self.get_logger().warn("couldn't find clusters")
            return

        self.get_logger().debug(f"{len(pcd.points)} points | {len(C)} clusters")
        if len(pcd.points) < 500:
            self.get_logger().warn("too few points after filtering!")
            return
        if len(C)==0:
            self.get_logger().warn("no clusters found")
            return
        
        #####
        ##### from here on, we're sure that we found clusters
        #####

        # get 3D points for each cluster
        Cp = [points[Ci] for Ci in C]

        # sort clusters by height and only select the 10 heighest clusters
        centers = np.array([np.mean(cp, axis=0) for cp in Cp])
        sorted_idxs = np.argsort(centers[:,HEIGHT_AXS])[::-1]

        centers = centers[sorted_idxs]
        centers = centers

        Cp = sort_list_by_indices(Cp, sorted_idxs)
        Cp = Cp[:10]

        C = sort_list_by_indices(C, sorted_idxs)
        C = C[:10]

        if centers[0,HEIGHT_AXS]<stack_center[HEIGHT_AXS]:
            print("didn't find any cluster higher than the stack center.")
            return
        
        grasp_point = centers[0].copy()
        # grasp_point[0] = stack_center[] # we want to grasp in the center of the stack, so we take the center of the whole stack instead of the cluster position

        # redo labels
        labels = np.zeros(len(points), dtype=np.int8)
        for l, idxs in enumerate(C):
            for i in idxs: labels[i] = l+1

        # colors are set based on the tab10 colorscheme, so the cluster order is easily visible in the visulization
        cmap = cm["tab10"]
        colors = np.array([cmap.colors[l-1] if l != 0 else [0,0,0] for i, l in enumerate(labels)]) # TODO this should be real color, not black

        # we select points near the cluster center to get a better height estimate (clusters tend to widen around the outer edges of the stack)
        tree = o3d.geometry.KDTreeFlann(pcd)
        [_, center_idxs, _] = tree.search_radius_vector_3d(grasp_point, 0.02)
        cluster_center_idxs = []
        for ci in center_idxs:
            if ci in C[0]: 
                cluster_center_idxs.append(ci)
                colors[ci] = [0,1,1]
        cluster_center_points = points[cluster_center_idxs]
        if len(cluster_center_idxs)==0:
            self.get_logger().warn("no grasp point cluster found")
            return

        cluster_center_size   = np.abs(np.max(cluster_center_points, axis=0) - np.min(cluster_center_points, axis=0))
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

        grasp_point[HEIGHT_AXS] = grasp_point[HEIGHT_AXS]-cluster_center_size[HEIGHT_AXS]/2 # the grasp point should be at the bottom of the center, so we subtract half it's height
        grasp_point[WIDTH_AXS] = stack_center[WIDTH_AXS]
        grasp_point[DEPTH_AXS] = stack_center[DEPTH_AXS]

        RT = R.T@tf.rotation_matrix(-np.pi, [0,0,1])[:3,:3]
        pcd = pcd.rotate(RT, center=[0,0,0])
        stack_center = RT@stack_center
        grasp_point = RT@grasp_point

        grasp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        grasp_sphere.translate(grasp_point)
        grasp_sphere.paint_uniform_color([1,0,0])
        geoms.append(grasp_sphere)

        center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        center_sphere.translate(stack_center)
        center_sphere.paint_uniform_color([0,0,1])
        geoms.append(center_sphere)
        
        # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
        # geoms += [pcd, mesh_frame]
        # o3d.visualization.draw_geometries(
        #     geoms, 
        #     point_show_normal=False,
        #     # **VIEW_PARAMS
        # )
        # exit()
        
        gp = PointStamped()

        gp.header.stamp = self.get_clock().now().to_msg()
        gp.header.frame_id = msg.header.frame_id
        gp.point.x = grasp_point[0]
        gp.point.y = grasp_point[1]
        gp.point.z = grasp_point[2]

        self.ppub.publish(gp)
        self.get_logger().debug(f"publishing point at {datetime.now()}")

        try:
            p_wrist = self.tf_buffer.transform(gp, "wrist_3_link")#, timeout=rclpy.duration.Duration(seconds=10))

            pose_wrist = PoseStamped()
            pose_wrist.header = p_wrist.header
            pose_wrist.pose.position = p_wrist.point
            pose_wrist.pose.position.x -= 0.01
            pose_wrist.pose.position.z -= 0.197
            self.posepub.publish(pose_wrist)

            self.get_logger().info(f"publishing pose at {datetime.now()}")
        except TransformException as ex:
            self.get_logger().warn(
                f'Could not transform msg.header.frame_id to world: {ex}')
        
        self.pcdpub.publish(convertCloudFromOpen3dToRos(pcd, frame_id=msg.header.frame_id))
        self.get_logger().debug(f"processing took {time.time()-start:2f}s", )


def main(args=None):
    """Creates subscriber node and spins it"""
    rclpy.init(args=args)

    executor = MultiThreadedExecutor(num_threads=8)
    node = StackDetector3D(executor=executor)

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
