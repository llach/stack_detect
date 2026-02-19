import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import JointState
import sys

class JointStatePrinter(Node):
    def __init__(self, group_name):
        super().__init__('joint_state_printer')
        
        self.group2joints = {
            "both": [
                "left_arm_shoulder_pan_joint", "left_arm_shoulder_lift_joint",
                "left_arm_elbow_joint", "left_arm_wrist_1_joint",
                "left_arm_wrist_2_joint", "left_arm_wrist_3_joint",
                "right_arm_shoulder_pan_joint", "right_arm_shoulder_lift_joint",
                "right_arm_elbow_joint", "right_arm_wrist_1_joint",
                "right_arm_wrist_2_joint", "right_arm_wrist_3_joint"
            ],
            "left": [
                "left_arm_shoulder_pan_joint", "left_arm_shoulder_lift_joint",
                "left_arm_elbow_joint", "left_arm_wrist_1_joint",
                "left_arm_wrist_2_joint", "left_arm_wrist_3_joint"
            ],
            "right": [
                "right_arm_shoulder_pan_joint", "right_arm_shoulder_lift_joint",
                "right_arm_elbow_joint", "right_arm_wrist_1_joint",
                "right_arm_wrist_2_joint", "right_arm_wrist_3_joint"
            ]
        }

        if group_name not in self.group2joints:
            self.get_logger().error(f"Invalid group: {group_name}. Use 'left', 'right', or 'both'.")
            sys.exit(1)

        self.target_group = group_name
        self.target_joints = self.group2joints[group_name]
        
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.listener_callback,
            10)
        
        self.get_logger().info(f"Waiting for joint states for group: {group_name}...")
        self.received = False

    def listener_callback(self, msg):
        if self.received:
            return

        # Create a mapping for quick lookup
        joint_map = dict(zip(msg.name, msg.position))
        
        # Check if all required joints are present in the message
        if all(joint in joint_map for joint in self.target_joints):
            ordered_positions = [joint_map[joint] for joint in self.target_joints]
            
            self.print_formatted_list(ordered_positions)
            self.received = True
            # Shutdown after printing to prevent spamming the terminal
            exit(0)

    def print_formatted_list(self, positions):
        print(f"\n# Current joint positions for: {self.target_group} [rad]")
        print("[")
        for pos in positions:
            # Formatting to 4 decimal places for cleanliness
            print(f"    {pos:.4f},")
        print("]\n")

        print(f"\n# Current joint positions for: {self.target_group} [deg]")
        print("[")
        for pos in positions:
            # Formatting to 4 decimal places for cleanliness
            print(f"    {np.rad2deg(pos):.2f},")
        print("]")


def main(args=None):
    rclpy.init(args=args)
    
    # Simple argument handling
    target = "both"
    if len(sys.argv) > 1:
        target = sys.argv[1].lower()
    else:
        print("Usage: ros2 run <pkg> <node> [left|right|both]")
        return

    node = JointStatePrinter(target)
    
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()