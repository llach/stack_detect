"""Subscriber module"""
import open3d as o3d
import rclpy
import numpy as np
import time

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

from tf2_ros import TransformBroadcaster
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points_numpy
import tf_transformations as tf

from matplotlib import colormaps as cm

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from moveit_msgs.srv import GetPositionIK

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

    # print(f"found {len(C)} cluster")
    for i, Ci in enumerate(C):
        for pi in Ci: labels[pi] = i+1
        # print(f"\t{i}: {len(Ci)}")

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

    # using all defaults
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=0.005))
    oboxes = pcd.detect_planar_patches(
        normal_variance_threshold_deg=60,
        coplanarity_deg=75,
        outlier_ratio=0.75,
        min_plane_edge_length=0,
        min_num_points=0,
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

    print("Detected {} patches".format(len(oboxes)))

    geometries = []
    for obox in oboxes:
        mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obox, scale=[1, 1, 0.0001])
        mesh.paint_uniform_color(obox.color)
        geometries.append(mesh)
        geometries.append(obox)
    geometries.append(pcd)

    # geoms = [inlier_cloud, outlier_cloud]
    o3d.visualization.draw_geometries(
        geometries, 
        point_show_normal=False,
        **VIEW_PARAMS
    )
    return

    # find the highest point in the plane (on whichever axis is up) and remove all points lower than that
    highest_plane_point = np.max(np.array(inlier_cloud.points)[:,0])
    higher_indices = np.where(outlier_pts[:,0]>highest_plane_point)[0]
    pcd = outlier_cloud.select_by_index(higher_indices)

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

        self.subscription = self.create_subscription(
            PointCloud2, "/camera/depth/color/points", self.point_cb, 0, callback_group=self.cb_group
        )

        self.tf_broadcaster = TransformBroadcaster(self)
        self.timer = self.create_timer(0.1, self.broadcast_timer_callback)#, executor=self.exe)

        self.ppub = self.create_publisher(PointStamped, '/grasp_point', 10, callback_group=self.cb_group)
        self.posepub = self.create_publisher(PoseStamped, '/grasp_pose', 10, callback_group=self.cb_group)

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
        pcd, _ = pcd.remove_radius_outlier(nb_points=250, radius=0.02)
        # pcd = remove_points_on_plane_and_below(pcd)

        # # step III: some pcds still had some noise, so we remove outliers based on density
        # pcd, _ = pcd.remove_radius_outlier(nb_points=250, radius=0.02)


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

        try:
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

                grasp_point = top_center.copy()
                grasp_point[1] = top_center[1]-Ccp_size[1]/2
                grasp_point = R@grasp_point

                cmap = cm["tab20"]
                colors = np.array([cmap.colors[l] if l != 0 else colors[i] for l in labels])
                for ci in Cc:
                    colors[ci] = [0, 1, 1]
        except:
            print("error")
            return
    
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
        # return
        if grasp_point is not None: 
            p = PoseStamped()

            p.header.stamp = self.get_clock().now().to_msg()
            p.header.frame_id = msg.header.frame_id
            p.pose.position.x = grasp_point[0]
            p.pose.position.y = grasp_point[1]
            p.pose.position.z = grasp_point[2]
            p.pose.orientation.x = 0.0
            p.pose.orientation.y = 0.0
            p.pose.orientation.z = 0.0
            p.pose.orientation.w = 1.0

            try:
                pwrist = self.tf_buffer.transform(p, "wrist_3_link", timeout=rclpy.duration.Duration(seconds=10))
                pwrist.pose.orientation.x = 0.0
                pwrist.pose.orientation.y = 0.0
                pwrist.pose.orientation.z = 0.0
                pwrist.pose.orientation.w = 1.0

                point_w = PointStamped()
                point_w.header = pwrist.header
                point_w.point = pwrist.pose.position
                self.ppub.publish(point_w)

                pwrist.pose.position.x -= 0.01
                pwrist.pose.position.z -= 0.197
                self.posepub.publish(pwrist)
                print("TF found")

            except TransformException as ex:
                self.get_logger().info(
                    f'Could not transform msg.header.frame_id to world: {ex}')
                return


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

    print("hi")
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
