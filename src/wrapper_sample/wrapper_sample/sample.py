import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Int32MultiArray

from wrapper_sample.other_package.robot import Robot


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(Int32MultiArray, 'state', 10)
        self.subscription = self.create_subscription(String, "gripper", self.gripper_callback, 10)

        self.timer = self.create_timer(0.5, self.timer_callback)
        self.robot = Robot("Iron Giant")

    def gripper_callback(self, msg):
        action = msg.data

        if action == "close":
            self.robot.close_gripper()
        elif action == "open":
            self.robot.open_gripper()

    def timer_callback(self):
        msg = Int32MultiArray()
        msg.data = self.robot.get_joints()
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
