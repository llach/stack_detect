from stack_msgs.srv import GripperService

import time
import rclpy
from rclpy.node import Node
import stack_approach.Engine_Activation_XM as xm

from stack_approach.robotiq_gripper import RobotiqGripper

class BlueGripperServiceClient(Node):

    def __init__(self):
        super().__init__('blue_gripper')
        self.srv = self.create_service(GripperService, 'gripper', self.srv_callback)

        self.portHandler, self.packetHandler = xm.open_port(4)
        xm.set_operating_mode([4,5], "position_mode", self.portHandler, self.packetHandler)
        xm.set_torque([4,5], True, self.portHandler, self.packetHandler)

        self.get_logger().info("setup done!")

    def close(self):
        xm.pos_control([5],[1912], self.portHandler, self.packetHandler) # Close gripper

    def open(self):
        xm.pos_control([5], [2940], self.portHandler, self.packetHandler) # Open gripper
            
    def srv_callback(self, request, response):
        self.open() if request.open else self.close()
        time.sleep(1.0)
        
        return response

def main():
    rclpy.init()

    minimal_service = BlueGripperServiceClient()

    rclpy.spin(minimal_service)
    rclpy.shutdown()


if __name__ == '__main__':
    main()