import rclpy
import numpy as np

from rcl_interfaces.srv import GetParameters
from control_msgs.action import FollowJointTrajectory
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from moveit_msgs.msg import DisplayRobotState
from tf2_geometry_msgs import PoseStamped
from geometry_msgs.msg import Pose, Point, Quaternion
from scipy.spatial.transform import Rotation as R

from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from controller_manager_msgs.srv import SwitchController, ListControllers

from std_msgs.msg import Header
from moveit_msgs.srv import GetPositionIK

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from stack_approach.helpers import call_cli_sync

class MotionHelper:
    TRAJ_CTRL = "scaled_joint_trajectory_controller"

    def __init__(self, node) -> None:
        self.n = node
        self.log = self.n.get_logger()

        self.current_q = None

        self.recbg = ReentrantCallbackGroup()
        self.mecbg = MutuallyExclusiveCallbackGroup()

        self.subscription = self.n.create_subscription(
            JointState, "/joint_states", self.js_callback, 0, callback_group=self.mecbg
        )

        self.statepub = self.n.create_publisher(DisplayRobotState, '/goal_state', 10, callback_group=self.recbg)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.n)

        # Set up service clients
        self.switch_controller_client = self.n.create_client(SwitchController, '/controller_manager/switch_controller')
        self.list_controllers_client = self.n.create_client(ListControllers, '/controller_manager/list_controllers')
        
        # Wait for the services to be available
        self.switch_controller_client.wait_for_service()
        self.list_controllers_client.wait_for_service()
        
        # Call to switch controller if needed
        self.check_and_switch_controller()

        self.traj_client = ActionClient(
            self.n, 
            FollowJointTrajectory, 
            f"/{self.TRAJ_CTRL}/follow_joint_trajectory",
            callback_group=self.recbg
        )

        self.log.info("waiting for trajectory action server ...")
        self.traj_client.wait_for_server()
        self.log.info("found action server!")

        self.log.info("getting controller joints ...")
        self.param_client = self.n.create_client(GetParameters, f"/{self.TRAJ_CTRL}/get_parameters")
        prm_future = self.param_client.call_async(GetParameters.Request(names=["joints"]))

        rclpy.spin_until_future_complete(self.n, prm_future)
        self.joint_names = prm_future.result().values[0].string_array_value

        self.log.info("got joints:")
        for j in self.joint_names: self.log.info(f"\t- {j}")

        self.ik_client = self.n.create_client(
            srv_type=GetPositionIK,
            srv_name="compute_ik",
            callback_group=self.recbg,
        )
        self.log.info("waiting for IK server ...")
        self.ik_client.wait_for_service()
        self.log.info("found IK server!")

        self.log.info("MH setup done")

    def check_and_switch_controller(self):
        # Check active controllers
        list_controllers_request = ListControllers.Request()
        future = self.list_controllers_client.call_async(list_controllers_request)
        rclpy.spin_until_future_complete(self.n, future)
        
        if future.result() is not None:
            controllers = future.result().controller
            active_controllers = [controller.name for controller in controllers if controller.state == 'active']
            
            self.n.get_logger().info(f"{active_controllers}")
            # Check if joint_trajectory_controller is active
            if 'joint_trajectory_controller' in active_controllers or "scaled_joint_trajectory_controller" not in active_controllers:
                self.n.get_logger().info("joint_trajectory_controller is active, switching to scaled_joint_trajectory_controller.")
                
                # Set up the request to switch controllers
                switch_request = SwitchController.Request()
                switch_request.deactivate_controllers = ['joint_trajectory_controller']
                switch_request.activate_controllers = ['scaled_joint_trajectory_controller']
                switch_request.strictness = SwitchController.Request.BEST_EFFORT

                # Call the switch_controller service
                switch_future = self.switch_controller_client.call_async(switch_request)
                rclpy.spin_until_future_complete(self.n, switch_future)
                
                if switch_future.result() is not None:
                    self.n.get_logger().info("Successfully switched to scaled_joint_trajectory_controller.")
                else:
                    self.n.get_logger().error("Failed to switch controllers.")
            else:
                self.n.get_logger().info("joint_trajectory_controller is not active, no switch needed.")
        else:
            self.n.get_logger().error("Failed to list controllers.")

    def moveit_IK(self, pose, state=None, ik_link="wrist_3_link", ntries=50):
        if state is None: state = self.current_q.copy()

        for i in range(ntries):
            self.n.get_logger().info(f"IK try {i}")
            ik_req = GetPositionIK.Request()
            ik_req.ik_request.group_name = "ur_manipulator"
            ik_req.ik_request.robot_state.joint_state.name = list(state.keys())
            ik_req.ik_request.robot_state.joint_state.position = list(state.values())
            ik_req.ik_request.ik_link_name = ik_link
            ik_req.ik_request.pose_stamped = pose

            res = self.ik_client.call(ik_req)

            if res.error_code.val != 1:
                # print(f"moveit error {res.error_code}")
                continue

            rs = DisplayRobotState()
            rs.state = res.solution
            self.statepub.publish(rs)

            return dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
        return None

    def js_callback(self, msg):
        self.current_q = {jname: q for jname, q in zip(msg.name, msg.position)}

    def send_traj_blocking(self, qfinal, t):
        print("sending goal")
        return self.traj_client.send_goal(
            self.create_traj(qfinal, t)
        )

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

    def get_wrist_pose(self):
        p = PoseStamped()
        p.header.stamp = rclpy.time.Time().to_msg()
        p.header.frame_id = "wrist_3_link"
        p.pose.position.x = 0.0
        p.pose.position.y = 0.0
        p.pose.position.z = 0.0
        p.pose.orientation.x = 0.0
        p.pose.orientation.y = 0.0
        p.pose.orientation.z = 0.0
        p.pose.orientation.w = 1.0
        return p
    
    def tf_msg_to_matrix(self, msg):
        t = msg.transform.translation
        t = [t.x, t.y, t.z]

        q = msg.transform.rotation
        q = [q.x, q.y, q.z, q.w]

        T = np.eye(4)
        T[:3,:3] = R.from_quat(q).as_matrix()
        T[:3,3] = t

        return T
    
    def matrix_to_pose_msg(self, T, frame_id):
        ps = PoseStamped(
            header=Header(
                stamp=rclpy.time.Time().to_msg(),
                frame_id=frame_id
            )
        )
        t = T[:3,3]
        q = R.from_matrix(T[:3,:3]).as_quat()
        ps.pose.position.x = t[0]
        ps.pose.position.y = t[1]
        ps.pose.position.z = t[2]

        ps.pose.orientation.x = q[0]
        ps.pose.orientation.y = q[1]
        ps.pose.orientation.z = q[2]
        ps.pose.orientation.w = q[3]
        
        return ps
    
    def inv(self, T):
        Ti = np.eye(4)
        Ti[:3,:3] = Ti[:3,:3].T
        Ti[:3,3] = -1*T[:3,3]
        return Ti
    
    def get_trafo(self, fro, to):
        return self.tf_msg_to_matrix(self.tf_buffer.lookup_transform(fro, to, rclpy.time.Time()))
    
    def move_rel(self, xyz, d, pose, secs):
        cs = [c for c in xyz.lower()]
        assert len(xyz)==1

        t = [0,0,0]
        for c in cs:
            c=c.lower()
            if c == "x": t[0] = d
            elif c == "y": t[1] = d
            elif c == "z": t[2] = d
            else:
                assert False, f"{c} not a valid dimension"

        pose.pose.position.x += t[0]
        pose.pose.position.y += t[1]
        pose.pose.position.z += t[2]

        qs = self.moveit_IK(pose)
        if qs is None:
            return False
        self.send_traj_blocking(qs, secs)
        return True
    
    def move_relative_wrist(self, xyz, dist, secs=5):
        d = dist
        for i in range(50):
            print(f"IK try {i} dist {d}")
            ret = self.move_rel(xyz, d, self.get_wrist_pose(), secs)
            if ret: return True

            d = np.random.uniform(dist-0.1*d, dist+0.1*d)

    
    def move_relative_map(self, xyz, dist, secs=5):
        d = dist
        for i in range(50):
            print(f"IK try {i} dist {d}")
            ret = self.move_rel(
                xyz, 
                d, 
                self.matrix_to_pose_msg(self.get_trafo("map", "wrist_3_link"), "map"),
                secs
            )
            if ret: return True

            d = np.random.uniform(dist-0.1*d, dist+0.1*d)