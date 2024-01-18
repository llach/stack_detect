"""Subscriber module"""
import rclpy
import threading    

import numpy as np
import tf_transformations as tf

from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rcl_interfaces.srv import GetParameters
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from std_msgs.msg import Int16MultiArray, Float64MultiArray
from sensor_msgs.msg import JointState
from stack_approach.robot_sim import RobotSim
from stack_approach.ik.common import trafo
from controller_manager import switch_controllers, list_controllers

class CtrlMode:
    TRAJ=0
    FRWRD=1

class MjViewer(Node):
    """Subscriber node"""

    ik_steps = 1
    px_gain = 0.002
    max_dt = 0.5
    max_dq = 0.01

    TRAJ_CTRL = "scaled_joint_trajectory_controller"
    FRWRD_CTRL = "forward_position_controller"

    def __init__(self, with_vis=True, rate=1, ctrl_mode=CtrlMode.TRAJ):
        super().__init__("mj_viewer")
        self.log = self.get_logger()

        self.with_vis = with_vis
        self.rate = self.create_rate(rate)
        self.ctrl_mode = ctrl_mode

        self.current_q = None

        self.cb_group = ReentrantCallbackGroup()

        # maximum age for messages
        self.last_update = rclpy.time.Time()
        self.max_age = rclpy.duration.Duration(seconds=self.max_dt)

        # set current image error to be uninitialized
        self.img_error = None

        self.rs = RobotSim(with_vis=with_vis)
        self.ur5 = self.rs.ur5

        self.subscription = self.create_subscription(
            JointState, "/joint_states", self.js_callback, 0
        )

        self.subscription = self.create_subscription(
            Int16MultiArray, "/line_img_error", self.err_cb, 0
        )

        if self.ctrl_mode == CtrlMode.FRWRD:
            self.qpub = self.create_publisher(Float64MultiArray, f'/{self.FRWRD_CTRL}/commands', 10)
            be_active = self.FRWRD_CTRL
            be_inactive = self.TRAJ_CTRL
        elif self.ctrl_mode == CtrlMode.TRAJ:
            self.traj_client = ActionClient(
                self, 
                FollowJointTrajectory, 
                "/scaled_joint_trajectory_controller/follow_joint_trajectory",
                callback_group=self.cb_group
            )
            self.current_goal = None

            be_active = self.TRAJ_CTRL 
            be_inactive = self.FRWRD_CTRL
        else:
            assert False, f"unknown control mode {self.ctrl_mode}"

        self.log.info("checking controllers ...")
        lc = list_controllers(self, "controller_manager")
        activate_controllers = []
        deactivate_controllers = []
        for c in lc.controller:
            if c.name in [self.TRAJ_CTRL, self.FRWRD_CTRL]: self.log.info(f"{c.name}: {c.state}")

            if c.name == be_active and c.state != "active":
                activate_controllers.append(be_active)

            if c.name == be_inactive and c.state == "active":
                deactivate_controllers.append(be_inactive)

        if len(activate_controllers) > 0 or len(deactivate_controllers) > 0:
            self.log.info(f"switching controllers\n\tactivate: {activate_controllers}\n\tdeactivate: {deactivate_controllers}")
            resp = switch_controllers(
                self,
                "controller_manager",
                activate_controllers=activate_controllers,
                deactivate_controllers=deactivate_controllers,
                strict=True,
                activate_asap=False,
                timeout=10.0
            )
            assert resp.ok, "could not switch controllers"
            self.log.info("switching successful!")

        # for the forwarding controller, ordering of the positions matter, so we query its parameter
        self.log.info("getting controller joints")

        self.param_client = self.create_client(GetParameters, f"/{be_active}/get_parameters")
        prm_future = self.param_client.call_async(GetParameters.Request(names=["joints"]))

        rclpy.spin_until_future_complete(self, prm_future)
        self.joint_names = prm_future.result().values[0].string_array_value

        self.log.info("got joints:")
        for j in self.joint_names: self.log.info(f"\t- {j}")

        self.log.info("setup done")

        self.timer = self.create_timer(1/rate, self.run, callback_group=self.cb_group)

    def js_callback(self, msg): 
        self.current_q = {jname: q for jname, q in zip(msg.name, msg.position)}

    def err_cb(self, msg):
        self.last_update = self.get_clock().now()
        self.img_error = msg.data

    def send_trajectory(self, q): pass

    def run(self):
        # if self.ctrl_mode == CtrlMode.TRAJ:
        #     self.log.info("waiting for trajectory action server")
        #     self.traj_client.wait_for_server()
        #     self.log.info("found action server!")

        if self.current_q is None: 
            self.log.warn("no current joint state")
            return

        # update robot state, robot model (self.ur5) is using this state for fk and ik!
        self.rs.update_robot_state(self.current_q)
        if self.with_vis: self.rs.render()

        #### 
        ####    Safety checks
        ####
        if self.img_error is None:
            self.log.warn("no image error received yet")
            return

        err_age = self.get_clock().now()-self.last_update
        if err_age > self.max_age:
            self.log.warn(f"data too old ({err_age} > {self.max_age})")
            return

        #### 
        ####    Motion generation
        ####
        T, J, Ts = self.ur5.fk(fk_type="space")
        (x_err, y_err) = self.img_error

        zGr = T[:3,:3]@[0,0,-1] # project z (or -z) in gripper frame
        zErr = np.arcsin( np.abs(np.dot(zGr, [0,0,1])) / (np.linalg.norm(zGr)*np.linalg.norm([0,0,1])))
                
        Toff = tf.rotation_matrix(-1*np.sign(zGr[2])*zErr, [1,0,0])


        # camGoal = T@trafo(t=[-self.px_gain*x_err,self.px_gain*y_err, 0])#@Toff
        camGoal = T@Toff

        if self.with_vis: self.rs.draw_goal(camGoal)

        # calculate IK. scaling orientation down improves accuracy in position
        qdot, N = self.ur5.ik([
            self.ur5.pose_task(camGoal, T, J, scale=(1, .6)),
        ])

        # clip deltaqs
        # qdot = np.clip(qdot, -self.max_dq, self.max_dq)
        qdot = self.ur5.extract_q_subset(qdot, self.joint_names)

        # calculate new qs, respect joint limits
        qnew = {n: np.clip(self.current_q[n]+qdot[n], -self.rs.qmax, self.rs.qmax) for n in self.joint_names}

        # print("now", self.current_q)
        # print("qd ", qdot)
        # print("new", qnew)
        # print()

        if self.ctrl_mode == CtrlMode.FRWRD: self.qpub.publish(Float64MultiArray(data=list(qnew.values())))
        elif self.ctrl_mode == CtrlMode.TRAJ: 
            traj_goal = FollowJointTrajectory.Goal()
            traj_goal.trajectory = JointTrajectory(
                joint_names=self.joint_names,
                points=[
                    JointTrajectoryPoint(
                        positions=list(qnew.values()),
                        time_from_start=rclpy.duration.Duration(seconds=10).to_msg()
                    )
                ]
            )
            self.current_goal = self.traj_client.send_goal_async(traj_goal)
        # except KeyboardInterrupt:
        #     print("shutting down ...")
        #     del self.rs
        #     if self.ctrl_mode == CtrlMode.TRAJ and self.current_goal is not None:
        #         print("cancelling goal ...")
        #         self.current_goal.cancel_goal()
        #         print("done")

def main(args=None):
    """Creates subscriber node and spins it"""
    rclpy.init(args=args)

    node = MjViewer()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        node.get_logger().info('Beginning client, shut down with CTRL-C')
        rclpy.spin(node, executor)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down.\n')
        node.timer.cancel()

        if node.ctrl_mode == CtrlMode.TRAJ and node.current_goal is not None:
                print("cancelling goal ...")
                rclpy.spin_until_future_complete(node, node.current_goal, executor)
                node.current_goal.result().cancel_goal_async()
                print("done")

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    if rclpy.ok(): rclpy.shutdown()


if __name__ == "__main__":
    main()
