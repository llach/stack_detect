"""Subscriber module"""
import rclpy
import time
from threading import Event

import threading 
from rcl_interfaces.srv import GetParameters
from control_msgs.action import FollowJointTrajectory
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from geometry_msgs.msg import TransformStamped, PointStamped, PoseStamped
from tf2_geometry_msgs import PoseStamped
from sensor_msgs.msg import JointState
from moveit_msgs.msg import DisplayRobotState

from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

import tf_transformations as tf

from moveit_msgs.srv import GetPositionIK

class StackGrasp(Node):
    """Subscriber node"""

    TRAJ_CTRL = "scaled_joint_trajectory_controller"

    def __init__(self, executor=None):
        self.exe = executor
        super().__init__("StackDetector3D")
        self.log = self.get_logger()

        self.cb_group = ReentrantCallbackGroup()

        self.current_q = None
        self.planning = False
        self.pwrist = None
        self.plt = self.create_timer(0.5, self.plan_timer, self.cb_group)

        self.poseusb = self.create_subscription(PoseStamped, '/grasp_pose', self.pose_cb, 0, callback_group=self.cb_group)

        self.subscription = self.create_subscription(
            JointState, "/joint_states", self.js_callback, 0, callback_group=self.cb_group
        )

        self.statepub = self.create_publisher(DisplayRobotState, '/goal_state', 10, callback_group=self.cb_group)

        self.traj_client = ActionClient(
            self, 
            FollowJointTrajectory, 
            "/scaled_joint_trajectory_controller/follow_joint_trajectory",
            callback_group=self.cb_group
        )

        self.log.info("waiting for trajectory action server")
        self.traj_client.wait_for_server()
        self.log.info("found action server!")

        self.log.info("getting controller joints")
        self.pl = threading.Lock()

        self.param_client = self.create_client(GetParameters, f"/{self.TRAJ_CTRL}/get_parameters")
        prm_future = self.param_client.call_async(GetParameters.Request(names=["joints"]))

        rclpy.spin_until_future_complete(self, prm_future)
        self.joint_names = prm_future.result().values[0].string_array_value

        self.ik_client = self.create_client(
            srv_type=GetPositionIK,
            srv_name="compute_ik",
            # callback_group=self.cb_group,
            callback_group=ReentrantCallbackGroup(),
        )
        self.log.info("waiting for IK server")
        self.traj_client.wait_for_server()
        self.log.info("found IK server!")

        self.log.info("got joints:")
        for j in self.joint_names: self.log.info(f"\t- {j}")

    def moveit_IK(self, state, pose, ik_link="wrist_3_link"):
        print("doing IK ... ")
        ik_req = GetPositionIK.Request()
        ik_req.ik_request.group_name = "ur_manipulator"
        ik_req.ik_request.robot_state.joint_state.name = list(state.keys())
        ik_req.ik_request.robot_state.joint_state.position = list(state.values())
        ik_req.ik_request.ik_link_name = ik_link
        ik_req.ik_request.pose_stamped = pose

        res = self.ik_client.call(ik_req)

        # self.exe.spin_until_future_complete(future, timeout_sec=3)
        # res = future.result()

        if res.error_code.val != 1:
            print(f"moveit error {res.error_code}")
            return None

        rs = DisplayRobotState()
        rs.state = res.solution
        self.statepub.publish(rs)

        print("IK done!")
        return dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))

    def plan_timer(self):
        if self.pwrist is None or self.current_q is None:
            print("not planning")
            return
    
        self.pl.acquire()
        
        print("approaching ...")
        approach_q = self.moveit_IK(self.current_q, self.pwrist)
        self.send_traj_blocking(approach_q, 7)

        print("inserting ...")
        pinsert = self.get_wrist_pose()
        pinsert.pose.position.z = -0.05

        insert_q = self.moveit_IK(approach_q, pinsert)
        self.send_traj_blocking(insert_q, 3)

        print("all done!")
        self.destroy_node()
        exit(0)

    def get_wrist_pose(self):
        p = PoseStamped()
        p.header.stamp = self.get_clock().now().to_msg()
        p.header.frame_id = "wrist_3_link"
        p.pose.position.x = 0.0
        p.pose.position.y = 0.0
        p.pose.position.z = 0.0
        p.pose.orientation.x = 0.0
        p.pose.orientation.y = 0.0
        p.pose.orientation.z = 0.0
        p.pose.orientation.w = 1.0
        return p

    def js_callback(self, msg): 
        self.current_q = {jname: q for jname, q in zip(msg.name, msg.position)}

    def send_traj_blocking(self, qfinal, t):
        fut = self.traj_client.send_goal_async(
            self.create_traj(qfinal, t)
        )
        rclpy.spin_until_future_complete(self, fut)
        time.sleep(t*1.05)
        return fut.result()

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

    def pose_cb(self, msg):
       if self.planning: return
       self.pwrist = msg

def moveit_IK(ik_client, state, pose, ik_link="wrist_3_link"):
    print("doing IK ... ")
    ik_req = GetPositionIK.Request()
    ik_req.ik_request.group_name = "ur_manipulator"
    ik_req.ik_request.robot_state.joint_state.name = list(state.keys())
    ik_req.ik_request.robot_state.joint_state.position = list(state.values())
    ik_req.ik_request.ik_link_name = ik_link
    ik_req.ik_request.pose_stamped = pose

    res = ik_client.call(ik_req)

    if res.error_code.val != 1:
        print(f"moveit error {res.error_code}")
        return None

    print("IK done!")
    return dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))



def main(args=None):
    """Creates subscriber node and spins it"""
    rclpy.init(args=args)

    curr_q = None
    def js_cb(msg):
        nonlocal curr_q
        curr_q = {jname: q for jname, q in zip(msg.name, msg.position)}

    curr_pose = None
    def pose_cb(msg):
        nonlocal curr_pose
        curr_pose = msg

    node = rclpy.create_node("mini")
    mt = MultiThreadedExecutor(num_threads=4)
    
    mt.add_node(node)

    cbg = ReentrantCallbackGroup()

    jssub = node.create_subscription(
        JointState, "/joint_states", js_cb, 0, callback_group=cbg
    )

    poseusb = node.create_subscription(PoseStamped, '/grasp_pose', pose_cb, 0, callback_group=cbg)

    ik_client = node.create_client(
        srv_type=GetPositionIK,
        srv_name="compute_ik",
        callback_group=ReentrantCallbackGroup(),
    )

    while curr_q is None:
        rclpy.spin_once(node)
    print("got js")
    while curr_pose is None:
        rclpy.spin_once(node)
    print("got pose")
    node.destroy_subscription(jssub)
    node.destroy_subscription(poseusb)

    print(curr_pose, curr_q)

    moveit_IK(ik_client, curr_q, curr_pose)

    # executor = MultiThreadedExecutor(num_threads=8)
    # node = StackGrasp(executor=executor)

    # executor.add_node(node)
    # executor.spin()

    # executor.add_node(node)

    # try:
    #     node.get_logger().info('Beginning client, shut down with CTRL-C')
    #     rclpy.spin(node, executor)
    # except KeyboardInterrupt:
    #     node.get_logger().info('Keyboard interrupt, shutting down.\n')

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
