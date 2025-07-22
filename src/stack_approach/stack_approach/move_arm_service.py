from stack_msgs.srv import MoveArm

import time
import rclpy
from rclpy.node import Node
from stack_approach.motion_helper import MotionHelper
from stack_approach.controller_switcher import ControllerSwitcher
from rclpy.executors import MultiThreadedExecutor


class MoveArmService(Node):

    def __init__(self):
        super().__init__('move_arm')

        self.srv = self.create_service(MoveArm, 'move_arm', self.srv_callback)
        self.mh = MotionHelper(self)
        self.controller_switcher = ControllerSwitcher()

        self.get_logger().info("setup done!")

    def srv_callback(self, request, response):
        self.controller_switcher.activate_controller(request.controller_name)

        if len(request.q_target) > 0:
            self.get_logger().info(f"executing q_target {request.q_target} ...")
            q_target = {jname: q for jname, q in zip(request.name_target, request.q_target)}
            self.mh.send_traj(q_target, request.execution_time, blocking=request.blocking)
            if request.blocking: time.sleep(0.5)
            self.get_logger().info("done")
            response.success = True
            return response

        q_start = self.mh.current_q.copy()
        self.get_logger().info(f"q_start {q_start}")
        self.get_logger().info(f"{request.target_pose}")
        if q_start is None:
            self.get_logger().error("No joint states yet!")
            response.success = False
            return response

        self.get_logger().info("doing IK ...")
        q_target = self.mh.moveit_IK(state=q_start, pose=request.target_pose, ik_link=request.ik_link)
        q_target = { jname: q_target[jname] for jname in request.name_target }
        self.get_logger().info(f"q_target {q_target}")

        if q_target is None:
            self.get_logger().error("IK error!")
            response.success = False
            return response
        self.get_logger().info("IK success!")
        
        if request.execute:
            self.get_logger().info("executing trajectory ...")
            self.mh.send_traj(q_target, request.execution_time, blocking=request.blocking)
            if request.blocking: time.sleep(0.5)
            self.get_logger().info("done")
        else:
            self.get_logger().info("execution not requested.")
            
        print(q_start, q_target)
        response.success = True
        response.name_start = list(q_start.keys())
        response.q_start = list(q_start.values())
        response.q_end = list(q_target.values())
        return response

def main():
    rclpy.init()

    minimal_service = MoveArmService()
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