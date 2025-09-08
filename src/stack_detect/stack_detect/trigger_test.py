#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


class TriggerTestServer(Node):
    def __init__(self):
        super().__init__('trigger_test_server')
        self.srv = self.create_service(Trigger, '/trigger_test', self.handle_trigger)
        self.get_logger().info('Trigger service [/trigger_test] ready')

    def handle_trigger(self, request, response):
        response.success = True
        response.message = "Trigger executed successfully!"
        self.get_logger().info('Service called -> returning success')
        return response


def main(args=None):
    rclpy.init(args=args)
    node = TriggerTestServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
