#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped

import rtde_receive
import numpy as np


class UR5RawFTPublisher(Node):

    def __init__(self):
        super().__init__('ur5_raw_ft_publisher')

        # ---------------- ROS ----------------
        self.publisher_ = self.create_publisher(
            WrenchStamped,
            '/ur5/ft_raw',
            10
        )

        # 150 Hz -> 1 / 150 ≈ 0.00667 s
        self.timer = self.create_timer(1.0 / 150.0, self.publish_ft)

        # ---------------- RTDE ----------------
        self.robot_ip = "192.168.56.101"
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)

        # ---------------- Bias estimation ----------------
        self.bias_samples_required = 100
        self.bias_buffer = []
        self.bias = np.zeros(6)
        self.bias_ready = False

        self.get_logger().info(
            f"Connected to UR RTDE at {self.robot_ip}"
        )
        self.get_logger().info(
            f"Collecting first {self.bias_samples_required} samples for bias calibration"
        )

    # ------------------------------------------------------------

    def publish_ft(self):
        """
        getActualTCPForce():
          [Fx, Fy, Fz, Tx, Ty, Tz]
          TCP frame, includes gravity
        """
        ft = self.rtde_r.getActualTCPForce()

        if ft is None or len(ft) != 6:
            return

        ft = np.array(ft)

        # ----- Bias estimation phase -----
        if not self.bias_ready:
            self.bias_buffer.append(ft)

            if len(self.bias_buffer) >= self.bias_samples_required:
                stacked = np.vstack(self.bias_buffer)
                self.bias[:] = np.mean(stacked, axis=0)
                self.bias_ready = True
                self.get_logger().info(
                    f"F/T bias calibrated: {self.bias}"
                )
            return

        # ----- Bias-corrected force -----
        ft_corr = ft - self.bias

        msg = WrenchStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "tool0"

        msg.wrench.force.x = float(ft_corr[0])
        msg.wrench.force.y = float(ft_corr[1])
        msg.wrench.force.z = float(ft_corr[2])
        msg.wrench.torque.x = float(ft_corr[3])
        msg.wrench.torque.y = float(ft_corr[4])
        msg.wrench.torque.z = float(ft_corr[5])

        self.publisher_.publish(msg)

    # ------------------------------------------------------------

    def destroy_node(self):
        self.get_logger().info("Shutting down RTDEReceiveInterface")
        self.rtde_r.disconnect()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = UR5RawFTPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
