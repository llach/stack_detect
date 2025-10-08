import time
import rclpy
from rclpy.node import Node
from softenable_display_msgs.srv import SetDisplay

class SetDisplaySwitcher(Node):
    def __init__(self):
        super().__init__('set_display_switcher')
        self.cli_display = self.create_client(SetDisplay, '/set_display')

        while not self.cli_display.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /set_display service...')

    def switch_displays(self):
        for i in range(10):
            preset = f"protocol_{i+1}"
            self.get_logger().info(f"Applying preset {preset}")

            req = SetDisplay.Request()
            req.name = preset
            req.use_tts = True

            future = self.cli_display.call_async(req)
            future.add_done_callback(self.response_callback)

            
            time.sleep(15)

    def response_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info(f"Service call succeeded: {response}")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = SetDisplaySwitcher()

    node.switch_displays()

    # Keep spinning to process callbacks until all futures are done
    rclpy.spin_once(node, timeout_sec=0.1)
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
