import os
import signal
import rclpy
import time
import numpy as np

import threading 
from rclpy.executors import MultiThreadedExecutor
from tf2_geometry_msgs import PoseStamped

from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from std_srvs.srv import Empty

import tf_transformations as tf
from geometry_msgs.msg import PointStamped

from stack_approach.motion_helper import MotionHelper
from scipy.spatial.transform import Rotation as R

class PrimitiveApproach(Node):

    def __init__(self, executor=None):
        self.exe = executor
        super().__init__("PrimitiveApproach")
        self.log = self.get_logger()

        self.planning = False
        self.pwrist = None

        self.pl = threading.Lock()
        self.mecbg = MutuallyExclusiveCallbackGroup()

        self.mh = MotionHelper(self, True)
        self.mh.gripper_pos(int(255*0.27))
        self.plt = self.create_service(Empty, "papp", self.approach)

        self.grasp_point = None
        self.gp_sub = self.create_subscription(PointStamped, '/grasp_point', self.gp_cb, 0)

    def gp_cb(self, msg):
        self.grasp_point = msg
        print("new point", self.grasp_point.header.frame_id, self.grasp_point.point)

    def approach(self, req, res):
        if self.mh.current_q is None:
            print("no joint state yet...")
            return res
        
        if self.mh.current_q is None:
            print("no joint state yet, waiting ...")
            return res
        
        # store start pose
        start_q = self.mh.current_q.copy()

        # set gripper to correct position
        self.mh.gripper_pos(int(255*0.27))
        
        self.approach_retries()

        tt = 1.5
        self.mh.move_relative_map("z", -0.002, secs=tt)
        self.mh.move_relative_map("y", -0.012, secs=tt)

        self.mh.move_relative_map("z", 0.009, secs=tt)
        self.mh.move_relative_map("y", -0.02, secs=tt)

        self.mh.close_gripper()
        time.sleep(1)

        self.mh.move_relative_map("z", 0.04, secs=2)
        self.mh.move_relative_map("y", 0.05, secs=2)

        time.sleep(1)
        self.mh.open_gripper()
        time.sleep(.5)

        self.mh.send_traj_blocking(start_q, 3)
        
        return res
    
    def approach_retries(self, orient=[180, 45, 90], ntries=100):
        dx, dy, dz = 0, 0, 0
        gp = self.grasp_point.point

        for i in range(ntries):
            gpl = [
                gp.x + dx, 
                gp.y + 0.02 + dy, 
                gp.z - 0.005 + dz
            ]
            
            print(f"{i} planning to {gpl}")
            Tmf = np.eye(4)
            Tmf[:3,3] = gpl
            Tmf[:3,:3] = R.from_euler("xyz", orient, degrees=True).as_matrix()

            pw_approach = self.finger_matrix_to_wrist_pose(Tmf)

            approach_q = self.mh.moveit_IK(pw_approach)
            if approach_q is not None: break
            dx = np.random.uniform(-0.045, 0.045)
            dy = np.random.uniform(-0.0005, 0.0005)
            dz = np.random.uniform(-0.0005, 0.0005)

        self.mh.send_traj_blocking(approach_q, 3)

    def grasp(self, req, res):
        if self.mh.current_q is None:
            print("no joint state yet...")
            return res
        
        if self.mh.current_q is None:
            print("no joint state yet, waiting ...")
            return res
        
        return res
    
    
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
    node = PrimitiveApproach(executor=executor)

    executor.add_node(node)
    executor.spin()

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    os.kill(os.getpid(), signal.SIGKILL)


if __name__ == "__main__":
    main()
