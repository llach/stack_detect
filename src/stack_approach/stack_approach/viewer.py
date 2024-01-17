"""Subscriber module"""
import rclpy
import threading    

import numpy as np
import tf_transformations as tf

from rclpy.node import Node
from std_msgs.msg import Int16MultiArray
from sensor_msgs.msg import JointState
from stack_approach.robot_sim import RobotSim
from stack_approach.ik.common import trafo

class MjViewer(Node):
    """Subscriber node"""

    ik_steps = 1
    px_gain = 0.002
    max_dt = 0.5
    max_dq = 0.005

    joint_names=[
        'shoulder_lift_joint', 
        'elbow_joint', 
        'wrist_1_joint', 
        'wrist_2_joint', 
        'wrist_3_joint', 
        'shoulder_pan_joint'
    ]

    def __init__(self, with_vis=True, rate=30):
        super().__init__("mj_viewer")
        self.log = self.get_logger()

        self.with_vis = with_vis
        self.rate = self.create_rate(rate)

        self.current_q = None

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

    def js_callback(self, msg): 
        self.current_q = {jname: q for jname, q in zip(msg.name, msg.position)}

    def err_cb(self, msg):
        self.last_update = self.get_clock().now()
        self.img_error = msg.data

    def run(self):
        while rclpy.ok():
            if self.current_q is None: continue

            # update robot state, robot model (self.ur5) is using this state for fk and ik!
            self.rs.update_robot_state(self.current_q)
            if self.with_vis: self.rs.render()

            #### 
            ####    Safety checks
            ####
            if self.img_error is None:
                print("no image error received yet")
                continue

            err_age = self.get_clock().now()-self.last_update
            if err_age > self.max_age:
                print(f"data too old ({err_age} > {self.max_age})")
                continue

            #### 
            ####    Motion generation
            ####

            T, J, Ts = self.ur5.fk(fk_type="space")
            (x_err, y_err) = self.img_error

            zGr = T[:3,:3]@[0,0,-1] # project z (or -z) in gripper frame
            zErr = np.arcsin( np.abs(np.dot(zGr, [0,0,1])) / (np.linalg.norm(zGr)*np.linalg.norm([0,0,1])))
                    
            Toff = tf.rotation_matrix(-1*np.sign(zGr[2])*zErr, [1,0,0])
            camGoal = T@trafo(t=[-self.px_gain*x_err,self.px_gain*y_err, 0])@Toff

            if self.with_vis: self.rs.draw_goal(camGoal)

            # calculate IK. scaling orientation down improves accuracy in position
            qdot, N = self.ur5.ik([
                self.ur5.pose_task(camGoal, T, J, scale=(1, .6)),
            ])

            # clip deltaqs
            qdot = np.clip(qdot, -self.max_dq, self.max_dq)
            qdot = self.ur5.extract_q_subset(qdot, self.joint_names)

            # calculate new qs
            qnew = {n: self.current_q[n]+qdot[n] for n in self.joint_names}
          
            self.rate.sleep()

def main(args=None):
    """Creates subscriber node and spins it"""
    rclpy.init(args=args)

    mj_viewer = MjViewer()

    thread = threading.Thread(target=rclpy.spin, args=(mj_viewer, ), daemon=True)
    thread.start()

    mj_viewer.run()

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    mj_viewer.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
