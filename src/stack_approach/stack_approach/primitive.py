import os
import signal
import rclpy
import time

import threading 
from rclpy.executors import MultiThreadedExecutor
from tf2_geometry_msgs import PoseStamped

from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

import tf_transformations as tf

from stack_approach.motion_helper import MotionHelper
from scipy.spatial.transform import Rotation as R

class Primitive(Node):
    """Subscriber node"""

    def __init__(self, executor=None):
        self.exe = executor
        super().__init__("Primitive")
        self.log = self.get_logger()

        self.planning = False
        self.pwrist = None

        self.pl = threading.Lock()
        self.mecbg = MutuallyExclusiveCallbackGroup()

        self.mh = MotionHelper(self, False)
        self.plt = self.create_timer(0.5, self.plan_timer, self.mecbg)

        self.wpub = self.create_publisher(PoseStamped, '/wrist_pose', 10)
        self.fpub = self.create_publisher(PoseStamped, '/finger_pose', 10)

    def plan_timer(self):
        if self.mh.current_q is None:
            print("no joint state yet, waiting ...")
            return
    
        # self.pl.acquire()
        # self.planning = True

        start_q = self.mh.current_q.copy()
        
        Tmf = self.mh.get_trafo("map", "finger")

        Tmf_approach = Tmf[:]
        Tmf_approach[:3,:3] = R.from_euler("xyz", [180, 55, 90], degrees=True).as_matrix()
        pw_approach = self.finger_matrix_to_wrist_pose(Tmf_approach)

        # approach_q = self.mh.moveit_IK(pw_approach)
        # self.mh.send_traj_blocking(approach_q, 1)

        self.mh.move_relative_map("z", [-0.005], secs=3)
        self.mh.move_relative_map("y", [-0.02], secs=3)
        self.mh.move_relative_map("z", [0.01], secs=3)
        self.mh.move_relative_map("y", [-0.02], secs=3)

        self.mh.send_traj_blocking(start_q, 3)
        
        # self.fpub.publish(self.mh.matrix_to_pose_msg(Tmf_approach, "map"))
        # self.wpub.publish(self.finger_matrix_to_wrist_pose(Tmf_approach))

    
        print("all done!")
        self.destroy_node()
        self.exe.shutdown()

    def finger_matrix_to_wrist_pose(self, Tf, frame_id="map"):
        Twf = self.mh.get_trafo("wrist_3_link", "finger")
        Tfw = self.mh.inv(Twf)
        return self.mh.matrix_to_pose_msg(Tf@Tfw, frame_id)

def main(args=None):
    """Creates subscriber node and spins it"""
    rclpy.init(args=args)

    executor = MultiThreadedExecutor(num_threads=4)
    node = Primitive(executor=executor)

    executor.add_node(node)
    executor.spin()

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    os.kill(os.getpid(), signal.SIGKILL)


if __name__ == "__main__":
    main()
