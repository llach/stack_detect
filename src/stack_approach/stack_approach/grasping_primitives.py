import time

from datetime import datetime

from stack_msgs.srv import MoveArm, GripperService
from stack_approach.helpers import call_cli_sync, empty_pose


def direct_approach_grasp(node, move_cli, gripper_cli, start_wrist_pose, with_grasp=True):
    """
    with_grasp: if False, only approach and retreat are executed. useful for finding grasp point offsets faster.
    """
    node.get_logger().info("opening gripper")
    gr = GripperService.Request()
    gr.open = True
    call_cli_sync(node, gripper_cli, gr)

    node.get_logger().info("Moving to grasp pose...")
    mr = MoveArm.Request()
    mr.execute = True
    mr.target_pose = start_wrist_pose
    approach_pose_res = call_cli_sync(node, move_cli, mr)
    print(approach_pose_res)
    
    if with_grasp:
        node.get_logger().info("inserting ...")
        pinsert = empty_pose(frame="wrist_3_link")
        pinsert.pose.position.z = 0.04

        mr = MoveArm.Request()
        mr.execute = True
        mr.target_pose = pinsert
        insert_pose_res = call_cli_sync(node, move_cli, mr)
        print(insert_pose_res)

        node.get_logger().info("closing gripper")
        gripper_close_time = datetime.now().timestamp()
        node.get_logger().info(f"{gripper_close_time}")
        gr = GripperService.Request()
        gr.open = False
        call_cli_sync(node, gripper_cli, gr)

        node.get_logger().info("lifting")
        plift = empty_pose(frame="wrist_3_link")
        plift.pose.position.x = 0.04

        mr = MoveArm.Request()
        mr.execute = True
        mr.target_pose = plift
        mr.execution_time = 0.5
        lift_pose_res = call_cli_sync(node, move_cli, mr)

        node.get_logger().info("retreating")
        pretr = empty_pose(frame="wrist_3_link")
        pretr.pose.position.z = -0.1

        mr = MoveArm.Request()
        mr.execute = True
        mr.target_pose = pretr
        mr.execution_time = 1.5
        retr_pose_res = call_cli_sync(node, move_cli, mr)
        print(retr_pose_res)
    else:
        time.sleep(5)

    node.get_logger().info("moving back to initial pose")
    mr = MoveArm.Request()
    mr.execute = True
    mr.name_target = approach_pose_res.name_start
    mr.q_target = approach_pose_res.q_start
    mr.execution_time = 2.0
    call_cli_sync(node, move_cli, mr)

    print("all done!")