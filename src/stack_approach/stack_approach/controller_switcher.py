import sys
import rclpy
from rclpy.node import Node
from controller_manager_msgs.srv import SwitchController, ListControllers

class ControllerSwitcher(Node):
    def __init__(self):
        super().__init__('auto_switch_controller')

        self.cli_switch = self.create_client(SwitchController, '/controller_manager/switch_controller')
        self.cli_list = self.create_client(ListControllers, '/controller_manager/list_controllers')

        while not self.cli_switch.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /controller_manager/switch_controller service...')
        while not self.cli_list.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /controller_manager/list_controllers service...')

        self.controllers = [
            "dual_arm_joint_trajectory_controller",
            "right_arm_joint_trajectory_controller",
            "left_arm_joint_trajectory_controller"
        ]

    def activate_controller(self, controller_name: str):
        if controller_name not in self.controllers:
            self.get_logger().error(f"'{controller_name}' is not a known controller.")
            return

        # Step 1: Get current controller states
        list_req = ListControllers.Request()
        future = self.cli_list.call_async(list_req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is None:
            self.get_logger().error('Failed to get controller list.')
            return

        active_controllers = [
            c.name for c in future.result().controller
            if c.state == 'active'
        ]

        if controller_name in active_controllers:
            print(f"{controller_name} is already active, nothing to do here!")
            return

        # Step 2: Check for conflicting active controllers
        to_deactivate = [
            c for c in self.controllers
            if c != controller_name and c in active_controllers
        ]

        self.get_logger().info(f"Will deactivate: {to_deactivate}")
        self.get_logger().info(f"Will activate: {controller_name}")

        # Step 3: Send switch request
        switch_req = SwitchController.Request()
        switch_req.activate_controllers = [controller_name]
        switch_req.deactivate_controllers = to_deactivate
        switch_req.strictness = SwitchController.Request.BEST_EFFORT

        future = self.cli_switch.call_async(switch_req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() and future.result().ok:
            self.get_logger().info(f"Successfully activated '{controller_name}' and deactivated conflicts.")
        else:
            self.get_logger().error(f"Failed to activate '{controller_name}'.")


def main(args=None):
    rclpy.init(args=args)
    node = ControllerSwitcher()

    if len(sys.argv) < 2:
        node.get_logger().error('Please provide the controller name as a command-line argument.')
        node.destroy_node()
        rclpy.shutdown()
        return

    controller_to_start = sys.argv[1]
    node.activate_controller(controller_to_start)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
