#!/usr/bin/env python3
import rclpy

from motion_helper_v2 import MotionHelperV2
from stack_msgs.srv import RollerGripper, RollerGripperV2


def force_open(node: MotionHelperV2, force_lim = 4):
    node.call_cli_sync(node.finger2srv["left_v2"], RollerGripperV2.Request(position=1.0))
    # node.call_cli_sync(node.finger2srv["left"], RollerGripper.Request(roller_vel=80, roller_duration=5.0))
    input("cont?")
    node.call_cli_sync(node.finger2srv["left_v2"], RollerGripperV2.Request(position=-1.0))
    node.open_gripper_on_force_change(force_lim)

def fc_down(node: MotionHelperV2, start_pose, force_goal=5):

    # ---- Go to start pose ----
    node.go_to_degrees(
        deg=start_pose,
        time=2,
        side="left"
    )

    node.ros_sleep(1.0)

    # ---- Wait for fresh F/T data ----
    node.get_logger().info("Waiting for fresh F/T data...")
    while rclpy.ok() and not node.has_fresh_ft():
        rclpy.spin_once(node, timeout_sec=0.1)

    node.get_logger().info("F/T data ready, starting descent")

    # ---- Force-controlled descent ----
    # success = node.descend_until_force(
    #     force_goal=force_goal,     # N
    #     delta_z=0.0004,     # m
    #     traj_time=0.02,     # s
    #     max_dist=0.03,      # m
    #     side="left"
    # )

    success = node.move_along_z_until_force_traj(
        force_goal=force_goal,     # N
        traj_time=3.0,     # s
        dist=-0.02,      # m
        side="left"
    )

    if success:
        node.get_logger().info("Contact detected successfully")
    else:
        node.get_logger().warn("Contact NOT detected")

    # input("cont?")
    
if __name__ == '__main__':
    rclpy.init()
    node = MotionHelperV2()

    fd = False

    # if fd:
        # node.call_cli_sync(node.finger2srv["left_v2"], RollerGripperV2.Request(position=0.75))
        # start_pose = [
        #     82.63,
        #     -92.23,
        #     -99.19,
        #     -82.52,
        #     -43.43,
        #     20.27
        # ]
        # fc_down(node, start_pose=start_pose, force_goal=45)

        # # node.call_cli_sync(node.finger2srv["left"], RollerGripper.Request(roller_vel=80, roller_duration=1.5))
        # # node.call_cli_sync(node.finger2srv["left_v2"], RollerGripperV2.Request(position=0.3))
        # node.call_cli_sync(node.finger2srv["left"], RollerGripper.Request(roller_vel=80, roller_duration=2.5))

        # # node.move_rel_wrist(
        # #     z=0.002,
        # #     time=1,
        # #     side="left"
        # # )
        # # node.call_cli_sync(node.finger2srv["left"], RollerGripper.Request(roller_vel=80, roller_duration=1.0))


        # node.call_cli_sync(node.finger2srv["left_v2"], RollerGripperV2.Request(position=-1.0))

        # # ---- Go to start pose ----
        # node.go_to_degrees(
        #     deg=start_pose,
        #     time=2,
        #     side="left"
        # )
  
        # node.call_cli_sync(node.finger2srv["left_v2"], RollerGripperV2.Request(position=0.5))

    # else:
    force_open(node, force_lim=3)

    rclpy.shutdown()