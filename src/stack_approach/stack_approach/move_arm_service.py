from stack_msgs.srv import MoveArm

import time
import rclpy
from rclpy.node import Node
from stack_approach.motion_helper import MotionHelper


class MoveArmService(Node):

    def __init__(self):
        super().__init__('move_arm')

        self.declare_parameter('with_gripper', True)
        self.with_gripper = self.get_parameter('with_gripper').get_parameter_value().bool_value

        self.srv = self.create_service(MoveArm, 'move_arm', self.srv_callback)
        self.mh = MotionHelper(self, self.with_gripper)

    def srv_callback(self, request, response):
        self.get_logger().info('Incoming request\na: ', request)
        
        print("doing IK ...")
        target_q = self.mh.moveit_IK(pose=request.target_pose, ik_link=request.ik_link)

        print("executing trajectory ...")
        self.send_traj_blocking(target_q, 3)
        time.sleep(2)
        print("done")
        
        return response

def main():
    rclpy.init()

    minimal_service = MoveArmService()

    rclpy.spin(minimal_service)
    rclpy.shutdown()


if __name__ == '__main__':
    main()