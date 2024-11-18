from stack_msgs.srv import GripperService

import time
import rclpy
from rclpy.node import Node

from stack_approach.robotiq_gripper import RobotiqGripper

class GripperServiceClient(Node):

    def __init__(self):
        super().__init__('gripper')

        self.declare_parameter('sim', True)
        self.sim = self.get_parameter('sim').get_parameter_value().bool_value

        self.srv = self.create_service(GripperService, 'gripper', self.srv_callback)
        if not self.sim:
            self.get_logger().info("gripper setup")
            self.gripper = RobotiqGripper()
            self.gripper.connect("192.168.56.101", 63352)
            self.gripper.activate(auto_calibrate=False)
            self.gripper.move_and_wait_for_pos(0, 0, 0)
        else:
            print("nothing to setup.")
        self.get_logger().info("setup done!")

    def close(self):
        self.gripper.move_and_wait_for_pos(255, 255, 255)

    def open(self):
        self.gripper.move_and_wait_for_pos(0, 0, 0)

    def gripper_pos(self, pos, vel=255, frc=255):
        return self.gripper.move_and_wait_for_pos(pos, vel, frc)
            
    def srv_callback(self, request, response):
        if self.sim:
            self.get_logger().info("not executing gripper action since we're in simulation")
            return response
        
        self.open() if request.open else self.close()
        time.sleep(1.5)
        
        return response

def main():
    rclpy.init()

    minimal_service = GripperServiceClient()

    rclpy.spin(minimal_service)
    rclpy.shutdown()


if __name__ == '__main__':
    main()