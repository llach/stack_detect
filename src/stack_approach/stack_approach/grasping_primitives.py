import time

from datetime import datetime

from stack_msgs.srv import MoveArm, GripperService
from stack_approach.roller_gripper import RollerGripper
from stack_approach.helpers import call_cli_sync, empty_pose, get_trafo, matrix_to_pose_msg


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
        pinsert.pose.position.z = 0.07

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
        print("sleeping")
        time.sleep(5)

    node.get_logger().info("moving back to initial pose")
    mr = MoveArm.Request()
    mr.execute = True
    mr.name_target = approach_pose_res.name_start
    mr.q_target = approach_pose_res.q_start
    mr.execution_time = 2.0
    call_cli_sync(node, move_cli, mr)

    print("all done!")
    
def angled_approach_grasp(node, move_cli, gripper_cli, start_wrist_pose, buffer, with_grasp=True):
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
    
    if with_grasp:
        
        ####
        ####        INSERT 1
        ####
        node.get_logger().info("inserting ...")
        Tmw = get_trafo("map", "wrist_3_link", buffer)
        Tmw[:3,3] += [0,0.02,0]
        pinsert = matrix_to_pose_msg(Tmw, "map")

        mr = MoveArm.Request()
        mr.execute = True
        mr.execution_time = 1.
        mr.target_pose = pinsert
        insert_pose_res = call_cli_sync(node, move_cli, mr)
        
        ####
        ####        INSERT 2
        ####
        Tmw = get_trafo("map", "wrist_3_link", buffer)
        Tmw[:3,3] += [0,0,0.005]
        pinsert2 = matrix_to_pose_msg(Tmw, "map")

        mr = MoveArm.Request()
        mr.execute = True
        mr.execution_time = 1.
        mr.target_pose = pinsert2
        insert_pose_res = call_cli_sync(node, move_cli, mr)
        
        ####
        ####        INSERT 3
        ####
        Tmw = get_trafo("map", "wrist_3_link", buffer)
        Tmw[:3,3] += [0,0.018,0]
        pinsert3 = matrix_to_pose_msg(Tmw, "map")

        mr = MoveArm.Request()
        mr.execute = True
        mr.execution_time = 1.
        mr.target_pose = pinsert3
        insert_pose_res = call_cli_sync(node, move_cli, mr)

        node.get_logger().info("closing gripper")
        gripper_close_time = datetime.now().timestamp()
        node.get_logger().info(f"{gripper_close_time}")
        gr = GripperService.Request()
        gr.open = False
        call_cli_sync(node, gripper_cli, gr)

        ####
        ####        LIFT
        ####
        Tmw = get_trafo("map", "wrist_3_link", buffer)
        Tmw[:3,3] += [0,0,0.03]
        pinsert3 = matrix_to_pose_msg(Tmw, "map")

        mr = MoveArm.Request()
        mr.execute = True
        mr.execution_time = 1.
        mr.target_pose = pinsert3
        insert_pose_res = call_cli_sync(node, move_cli, mr)
        
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

def dark_stack_roller(node, move_cli, gripper_cli, start_wrist_pose, buffer, with_grasp=True):
    """
    with_grasp: if False, only approach and retreat are executed. useful for finding grasp point offsets faster.
    """
    roller = RollerGripper()
    roller.open()

    # node.get_logger().info("opening gripper")
    # gr = GripperService.Request()
    # gr.open = True
    # call_cli_sync(node, gripper_cli, gr)


    node.get_logger().info("Moving to grasp pose...")
    mr = MoveArm.Request()
    mr.execute = True
    mr.target_pose = start_wrist_pose
    approach_pose_res = call_cli_sync(node, move_cli, mr)
    
    if with_grasp:
        
        ####
        ####        INSERT 1
        ####
        # node.get_logger().info("inserting ...")
        # Tmw = get_trafo("map", "wrist_3_link", buffer)
        # Tmw[:3,3] += [0,0.0,0.005]
        # pinsert = matrix_to_pose_msg(Tmw, "map")

        # mr = MoveArm.Request()
        # mr.execute = True
        # mr.execution_time = 1.
        # mr.target_pose = pinsert
        # insert_pose_res = call_cli_sync(node, move_cli, mr)
        
        ####
        ####        INSERT 2
        ####
        Tmw = get_trafo("map", "wrist_3_link", buffer)
        Tmw[:3,3] += [0,0.035,0]
        pinsert2 = matrix_to_pose_msg(Tmw, "map")

        mr = MoveArm.Request()
        mr.execute = True
        mr.execution_time = 1.
        mr.target_pose = pinsert2
        insert_pose_res = call_cli_sync(node, move_cli, mr)
        
        # ####
        # ####        INSERT 3
        # ####
        # Tmw = get_trafo("map", "wrist_3_link", buffer)
        # Tmw[:3,3] += [0,0,-0.015]
        # pinsert3 = matrix_to_pose_msg(Tmw, "map")

        # mr = MoveArm.Request()
        # mr.execute = True
        # mr.execution_time = 1.
        # mr.target_pose = pinsert3
        # insert_pose_res = call_cli_sync(node, move_cli, mr)

        # node.get_logger().info("closing gripper a bit")
        # gripper_close_time = datetime.now().timestamp()
        # node.get_logger().info(f"{gripper_close_time}")
        # gr = GripperService.Request()
        # gr.open = True
        # gr.opening = 80
        # call_cli_sync(node, gripper_cli, gr)

        # roller.roll(3)

        # node.get_logger().info("closing gripper")
        # gripper_close_time = datetime.now().timestamp()
        # node.get_logger().info(f"{gripper_close_time}")
        # gr = GripperService.Request()
        # gr.open = False
        # call_cli_sync(node, gripper_cli, gr)
        roller.close()


        # ####
        # ####        LIFT
        # ####
        Tmw = get_trafo("map", "wrist_3_link", buffer)
        Tmw[:3,3] += [0,0,0.03]
        pinsert3 = matrix_to_pose_msg(Tmw, "map")

        mr = MoveArm.Request()
        mr.execute = True
        mr.execution_time = 1.
        mr.target_pose = pinsert3
        insert_pose_res = call_cli_sync(node, move_cli, mr)
        
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

def thin_stack_roller(node, move_cli, gripper_cli, start_wrist_pose, buffer, with_grasp=True):
    """
    with_grasp: if False, only approach and retreat are executed. useful for finding grasp point offsets faster.
    """
    roller = RollerGripper()

    node.get_logger().info("opening gripper")
    gr = GripperService.Request()
    gr.open = True
    call_cli_sync(node, gripper_cli, gr)

    node.get_logger().info("Moving to grasp pose...")
    mr = MoveArm.Request()
    mr.execute = True
    mr.target_pose = start_wrist_pose
    approach_pose_res = call_cli_sync(node, move_cli, mr)
    
    if with_grasp:
        
        ####
        ####        INSERT 1
        ####
        node.get_logger().info("inserting ...")
        Tmw = get_trafo("map", "wrist_3_link", buffer)
        Tmw[:3,3] += [0,0.0,0.002]
        pinsert = matrix_to_pose_msg(Tmw, "map")

        mr = MoveArm.Request()
        mr.execute = True
        mr.execution_time = 1.
        mr.target_pose = pinsert
        insert_pose_res = call_cli_sync(node, move_cli, mr)
        
        ####
        ####        INSERT 2
        ####
        Tmw = get_trafo("map", "wrist_3_link", buffer)
        Tmw[:3,3] += [0,0.015,0]
        pinsert2 = matrix_to_pose_msg(Tmw, "map")

        mr = MoveArm.Request()
        mr.execute = True
        mr.execution_time = 1.
        mr.target_pose = pinsert2
        insert_pose_res = call_cli_sync(node, move_cli, mr)
        
        # ####
        # ####        INSERT 3
        # ####
        Tmw = get_trafo("map", "wrist_3_link", buffer)
        Tmw[:3,3] += [0,0,-0.015]
        pinsert3 = matrix_to_pose_msg(Tmw, "map")

        mr = MoveArm.Request()
        mr.execute = True
        mr.execution_time = 1.
        mr.target_pose = pinsert3
        insert_pose_res = call_cli_sync(node, move_cli, mr)

        node.get_logger().info("closing gripper a bit")
        gripper_close_time = datetime.now().timestamp()
        node.get_logger().info(f"{gripper_close_time}")
        gr = GripperService.Request()
        gr.open = True
        gr.opening = 150
        call_cli_sync(node, gripper_cli, gr)

        roller.roll(3)

        # node.get_logger().info("closing gripper")
        # gripper_close_time = datetime.now().timestamp()
        # node.get_logger().info(f"{gripper_close_time}")
        # gr = GripperService.Request()
        # gr.open = False
        # call_cli_sync(node, gripper_cli, gr)

        # ####
        # ####        LIFT
        # ####
        # Tmw = get_trafo("map", "wrist_3_link", buffer)
        # Tmw[:3,3] += [0,0,0.03]
        # pinsert3 = matrix_to_pose_msg(Tmw, "map")

        # mr = MoveArm.Request()
        # mr.execute = True
        # mr.execution_time = 1.
        # mr.target_pose = pinsert3
        # insert_pose_res = call_cli_sync(node, move_cli, mr)
        
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