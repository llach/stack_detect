"""Subscriber module"""
import rclpy

from rclpy.node import Node
import threading    

from sensor_msgs.msg import JointState
from stack_approach.robot_sim import RobotSim
from stack_approach.ik.common import mj2quat

class MjViewer(Node):
    """Subscriber node"""

    ik_steps = 1
    px_gain = 0.002
    max_dt = 1

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
        self.with_vis = with_vis
        self.rate = self.create_rate(rate)

        self.current_q = {n: 0 for n in self.joint_names}
        self.last_update = rclpy.time.Time()
        self.max_age = rclpy.duration.Duration(seconds=self.max_dt)

        self.rs = RobotSim(with_vis=with_vis)

        self.subscription = self.create_subscription(
            JointState, "/joint_states", self.js_callback, 0
        )

    def js_callback(self, msg):
        "update latest time and current joint values"
        self.last_update = rclpy.time.Time.from_msg(msg.header.stamp)
        for jname, q in zip(msg.name, msg.position): self.current_q[jname]=q

    def run(self):
        while rclpy.ok():
            if self.with_vis: self.rs.step(self.current_q)

            # print(self.get_clock().now()-self.last_update < self.max_age)            
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
