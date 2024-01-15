"""Subscriber module"""
import rclpy
from rclpy.node import Node

from std_msgs.msg import String


class TowelDetector(Node):
    """Subscriber node"""

    def __init__(self):
        super().__init__("minimal_subscriber")
        self.subscription = self.create_subscription(
            String, "topic", self.listener_callback, 10
        )

    def listener_callback(self, msg):
        """Callback function when message is received"""
        self.get_logger().info(f'I heard: "{msg.data}"')


def main(args=None):
    """Creates subscriber node and spins it"""
    rclpy.init(args=args)

    minimal_subscriber = TowelDetector()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
