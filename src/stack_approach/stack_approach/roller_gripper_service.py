import time
import rclpy

from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

import stack_approach.Engine_Activation_XM as xm
from stack_msgs.srv import RollerGripper

CAMERA_FINGER_PORT = 1
CAMERA_ROLLER_PORT = 2
NORMAL_ROLLER_PORT = 3
NORMAL_FINGER_PORT = 4

class RollerGripperService(Node):
    def __init__(self):
        super().__init__('roller_gripper')

        self.declare_parameter("finger_port", 1)
        self.declare_parameter("roller_port", 2)

        self.FINGER_PORT = self.get_parameter('finger_port').get_parameter_value().integer_value
        self.ROLLER_PORT = self.get_parameter('roller_port').get_parameter_value().integer_value

        assert self.FINGER_PORT in [CAMERA_FINGER_PORT, NORMAL_FINGER_PORT], f"unknown finger port {self.FINGER_PORT}"
        assert self.ROLLER_PORT in [CAMERA_ROLLER_PORT, NORMAL_ROLLER_PORT], f"unknown roller port {self.ROLLER_PORT}"

        self.SRV_PREFIX = "right" if self.FINGER_PORT == CAMERA_FINGER_PORT else "left"
        
        self.srv = self.create_service(RollerGripper, f'{self.SRV_PREFIX}_roller_gripper', self.srv_callback)

        self.portHandler, self.packetHandler = xm.open_port_by_device(self.FINGER_PORT, f"/dev/dynamixel_{self.SRV_PREFIX}")

        xm.set_operating_mode([self.FINGER_PORT], "position_mode", self.portHandler, self.packetHandler)
        xm.set_operating_mode([self.ROLLER_PORT], "velocity_mode", self.portHandler, self.packetHandler)

        xm.set_torque([self.ROLLER_PORT], True, self.portHandler, self.packetHandler)
        xm.set_torque([self.FINGER_PORT], True, self.portHandler, self.packetHandler)

        self.get_logger().info("setup done!")

    def srv_callback(self, req, res):
        self.get_logger().info(f"got gripper request\n{req}")

        if req.finger_pos != -1:
            self.get_logger().info(f"moving finger to {req.finger_pos}")
            if self.FINGER_PORT == CAMERA_FINGER_PORT:
                assert 700 <= req.finger_pos <= 2500, f"invalid finger position {req.finger_pos}"
            elif self.FINGER_PORT == NORMAL_FINGER_PORT:
                assert 2000 <= req.finger_pos <= 3500, f"invalid finger position {req.finger_pos}"
            xm.pos_control([self.FINGER_PORT],[req.finger_pos], self.portHandler, self.packetHandler)
            time.sleep(.5)
        elif req.roller_duration != -1:
            self.get_logger().info(f"rolling at vel {req.roller_vel} for {req.roller_duration}")

            assert -100 <= req.roller_vel <= 100, f"invalid roller velocity {req.roller_vel}" 
            xm.vel_control([self.ROLLER_PORT],[req.roller_vel], self.portHandler, self.packetHandler)
            time.sleep(req.roller_duration)
            xm.vel_control([self.ROLLER_PORT],[0], self.portHandler, self.packetHandler)
        else:
            print("malformed request")
            res.success = False
            return res

        self.get_logger().info("request done")


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