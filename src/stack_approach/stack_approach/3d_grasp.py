"""Subscriber module"""
import open3d as o3d
import rclpy
import collections
import numpy as np
import ros2_numpy as rnp
import ctypes
import struct
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import TransformStamped, PointStamped

from tf2_ros import TransformBroadcaster
from rclpy.node import Node
from cv_bridge import CvBridge 
from rclpy.parameter import Parameter
from rclpy.callback_groups import ReentrantCallbackGroup

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points_numpy
from rcl_interfaces.msg import SetParametersResult
import tf_transformations as tf

from matplotlib import colormaps as cm

"""
{
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 0.10000000000000002, 0.14327596127986908, 0.43650001287460327 ],
			"boundingbox_min" : [ -0.045445997267961502, -0.14484645426273346, -0.0060000000000000001 ],
			"field_of_view" : 60.0,
			"front" : [ 0.12346865912958778, -0.20876634334886676, -0.97014024970489954 ],
			"lookat" : [ 0.012304816534506125, -0.0072898239635080433, 0.21474424582004883 ],
			"up" : [ 0.9914498309557469, 0.06754698721022645, 0.11164513969108841 ],
			"zoom" : 0.35999999999999965
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}

"""

VIEW_PARAMS = { 
    "front" : [ 0.12346865912958778, -0.20876634334886676, -0.97014024970489954 ],
    "lookat" : [ 0.012304816534506125, -0.0072898239635080433, 0.21474424582004883 ],
    "up" : [ 0.9914498309557469, 0.06754698721022645, 0.11164513969108841 ],
    "zoom" : 0.35999999999999965
}

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

    print(f"found {len(C)} cluster")
    for i, Ci in enumerate(C):
        for pi in Ci: labels[pi] = i+1
        print(f"\t{i}: {len(Ci)}")

    return C, labels


def remove_points_on_plane_and_below(pcd):
    _, inliers = pcd.segment_plane(distance_threshold=0.01,
                                            ransac_n=3,
                                            num_iterations=1000)

    # get inliers and outliers, paint for debugging
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    outlier_pts = np.array(outlier_cloud.points)

    # find the highest point in the plane (on whichever axis is up) and remove all points lower than that
    highest_plane_point = np.max(np.array(inlier_cloud.points)[:,0])
    higher_indices = np.where(outlier_pts[:,0]>highest_plane_point)[0]
    pcd = outlier_cloud.select_by_index(higher_indices)

    return pcd


class StackDetector3D(Node):
    """Subscriber node"""

    def __init__(self, executor):
        self.exe = executor
        super().__init__("StackDetector3D")

        self.cb_group = ReentrantCallbackGroup()

        self.subscription = self.create_subscription(
            PointCloud2, "/camera/depth/color/points", self.point_cb, 0, callback_group=self.cb_group
        )

        self.tf_broadcaster = TransformBroadcaster(self)
        self.timer = self.create_timer(0.1, self.broadcast_timer_callback)#, executor=self.exe)

        self.ppub = self.create_publisher(PointStamped, '/grasp_point', 10, callback_group=self.cb_group)

    def broadcast_timer_callback(self):
       t = TransformStamped()

       t.header.stamp = self.get_clock().now().to_msg()
       t.header.frame_id = 'hande_right_finger'
       t.child_frame_id = 'tip_frame'
       t.transform.translation.x = 0.0
       t.transform.translation.y = 0.0
       t.transform.translation.z = 0.05
       t.transform.rotation.x = 0.0
       t.transform.rotation.y = 0.0
       t.transform.rotation.z = 0.0
       t.transform.rotation.w = 1.0

       self.tf_broadcaster.sendTransform(t)

    def publish_point(self, gp, frame):
        p = PointStamped()

        p.header.stamp = self.get_clock().now().to_msg()
        p.header.frame_id = frame
        p.point.x = gp[0]
        p.point.y = gp[1]
        p.point.z = gp[2]

        self.ppub.publish(p)

    def point_cb(self, msg):
        points = read_points_numpy(msg)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        pcd.paint_uniform_color([0,0,0])

        # step I: crop cloud
        bb_min, bb_max = [-0.2,-0.25,0], [0.2, 0.25, 0.6]
        aabb = o3d.geometry.AxisAlignedBoundingBox(bb_min, bb_max)
        aabb.color = [1,0,0]

        pcd = pcd.crop(aabb)

        # step II: remove supporting plane and points below
        pcd = remove_points_on_plane_and_below(pcd)

        # step III: some pcds still had some noise, so we remove outliers based on density
        pcd, _ = pcd.remove_radius_outlier(nb_points=50, radius=0.02)


        Rz = tf.rotation_matrix(np.pi/2, [0,0,1])[:3,:3]
        Ry = tf.rotation_matrix(np.pi, [0,1,0])[:3,:3]
        R = Ry@Rz

        pcd = pcd.rotate(R, center=[0,0,0])

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=0.005))
        nls = np.array(pcd.normals)[:]

        # project normals to xz-plane
        angs = np.clip(np.arctan2(nls[:,1], nls[:,2]), -np.pi/2, np.pi/2)+(np.pi/2)
        angs /= np.pi

        sms = cm["Spectral"]
        colors = np.array(sms(angs))

        colors = np.array(pcd.colors)
        colors_angles = sms(angs)[:,:3]
        for i, ang in enumerate(angs):
            if ang < 0.4 and ang > 0.1: colors[i] = colors_angles[i]

        points = np.array(pcd.points)
        stack_center = np.mean(points, axis=0)
       
        C, labels = random_growing_clusters(pcd, np.all([angs<0.4, angs>0.1], axis=0), k=20, min_size=50)
        grasp_point = None

        if len(C)>0:
            Cp = [points[Ci] for Ci in C]
            centers = np.array([np.mean(cp, axis=0) for cp in Cp])
            centers = centers[centers[:,1]>stack_center[1]] # filter by height

            if len(centers)==0:
                print("no clusters left after height filtering")
                return

            top_clust_idx = np.argmax(centers[:,1])
            top_center = centers[top_clust_idx]

            tree = o3d.geometry.KDTreeFlann(pcd)
            [_, center_idxs, _] = tree.search_radius_vector_3d(top_center, 0.02)
            Cc = []
            for ci in center_idxs:
                if ci in C[top_clust_idx]: Cc.append(ci)

            Ccp = points[Cc]
            if len(Ccp)==0:
                print("no clusters found")
                return
        
            Ccp_size = np.abs(np.max(Ccp, axis=0)-np.min(Ccp, axis=0))

            grasp_point = stack_center.copy()
            grasp_point[1] = top_center[1]-Ccp_size[1]/2
            grasp_point = R@grasp_point

            cmap = cm["tab20"]
            colors = np.array([cmap.colors[l] if l != 0 else colors[i] for l in labels])
            for ci in Cc:
                colors[ci] = [0, 1, 1]

        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        pcd = pcd.rotate(R.T, center=[0,0,0])
        stack_center = R@stack_center

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
        geoms = [pcd, mesh_frame]

        if grasp_point is not None:
            grasp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
            grasp_sphere.translate(grasp_point)
            grasp_sphere.paint_uniform_color([1,0,0])
            geoms.append(grasp_sphere)

        center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        center_sphere.translate(stack_center)
        center_sphere.paint_uniform_color([0,0,1])
        geoms.append(center_sphere)
        
        # o3d.visualization.draw_geometries(
        #     geoms, 
        #     point_show_normal=False,
        #     **VIEW_PARAMS
        # )
        self.publish_point(grasp_point, msg.header.frame_id)


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
