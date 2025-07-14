import time
import rclpy

from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

import stack_approach.Engine_Activation_XM as xm
from stack_msgs.srv import RollerGripper

class RollerGripperService(Node):
    LEFT_FINGER_PORT = 1
    LEFT_ROLLER_PORT = 2

    def __init__(self):
        super().__init__('roller_gripper')

        self.srv = self.create_service(RollerGripper, 'roller_gripper', self.srv_callback)

        self.portHandler, self.packetHandler = xm.open_port(self.LEFT_FINGER_PORT)

        xm.set_operating_mode([self.LEFT_FINGER_PORT], "position_mode", self.portHandler, self.packetHandler)
        xm.set_operating_mode([self.LEFT_ROLLER_PORT], "velocity_mode", self.portHandler, self.packetHandler)

        xm.set_torque([self.LEFT_ROLLER_PORT], True, self.portHandler, self.packetHandler)
        xm.set_torque([self.LEFT_FINGER_PORT], True, self.portHandler, self.packetHandler)

        self.get_logger().info("setup done!")

    def srv_callback(self, req, res):
        if req.finger_pos != -1:
            assert 700 <= req.finger_pos <= 2500, f"invalid finger position {req.finger_pos}" 
            xm.pos_control([self.LEFT_FINGER_PORT],[req.finger_pos], self.portHandler, self.packetHandler)
            time.sleep(.5)
        elif req.roller_duration != -1:
            assert -100 <= req.roller_vel <= 100, f"invalid roller velocity {req.roller_vel}" 
            xm.vel_control([self.LEFT_ROLLER_PORT],[req.roller_vel], self.portHandler, self.packetHandler)
            time.sleep(req.roller_duration)
            xm.vel_control([self.LEFT_ROLLER_PORT],[0], self.portHandler, self.packetHandler)
        else:
            print("malformed request")
            res.success = False
            return res

        self.get_logger().info("got gripper request")
        print(req)

        res.success = True
        return res
    
def main():
    rclpy.init()

    minimal_service = RollerGripperService()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(minimal_service)

    try:
        executor.spin()
    except KeyboardInterrupt:
        print("Keyboard Interrupt!")
    finally:
        minimal_service.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()