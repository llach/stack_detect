import os
import rclpy
import time
import json
from datetime import datetime

import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from tf2_ros import TransformListener, Buffer

from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from tf2_geometry_msgs import PoseStamped

from stack_msgs.srv import CloudPose, CloudPoseVary, StoreData, MoveArm, GripperService
from stack_msgs.action import RecordCloud
from stack_approach.helpers import pose_to_list, call_cli_sync, empty_pose, transform_to_pose_stamped


class DataCollectionActionClient(Node):
    def __init__(self, store_dir = f"{os.environ['HOME']}/repos/unstack_cloud"):
        super().__init__('data_collection_action_client')
        self.store_dir = store_dir

        self.declare_parameter('sim', False)
        self.declare_parameter('min_samples', 30)
        self.declare_parameter('sampling_radius', 0.001)
        self.declare_parameter('offset_interval', 0.02)

        self.sim = self.get_parameter("sim").get_parameter_value().bool_value
        self.min_samples = self.get_parameter('min_samples').get_parameter_value().integer_value
        self.sampling_radius = self.get_parameter('sampling_radius').get_parameter_value().double_value
        self.offset_interval = self.get_parameter('offset_interval').get_parameter_value().double_value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pose_array_publisher = self.create_publisher(PoseArray, '/record_poses', 10)

        self._action_client = ActionClient(self, RecordCloud, 'collect_cloud_data', callback_group=ReentrantCallbackGroup())
        while not self._action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('data collection action not available, waiting again...')

        self.pose_cli = self.create_client(CloudPose, 'cloud_pose')
        while not self.pose_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('pose service not available, waiting again...')

        self.vary_pose_cli = self.create_client(CloudPoseVary, 'cloud_pose_vary')
        while not self.vary_pose_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('vary pose service not available, waiting again...')

        self.store_cli = self.create_client(StoreData, "store_cloud_data")
        while not self.store_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('store data service not available, waiting again...')

        self.move_cli = self.create_client(MoveArm, "move_arm")
        while not self.move_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('move arm service not available, waiting again...')

        self.gripper_cli = self.create_client(GripperService, "gripper")
        while not self.gripper_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('gripper service not available, waiting again...')

        self.n_samples = np.array([0])

        self.get_logger().info('setup done!')


    def start_recording(self):
        # Create a goal to start data collection
        goal_msg = RecordCloud.Goal()
        goal_msg.start = True  # Set the start flag

        # Send the goal
        self.get_logger().info('Sending goal to start data collection...')
        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        rclpy.spin_until_future_complete(self, send_goal_future)

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().fatal('Goal rejected by action server.')
            exit(0)

        self.get_logger().info('Goal accepted by action server.')
        return goal_handle

    def feedback_callback(self, feedback_msg):
        self.n_samples = np.array(feedback_msg.feedback.n_samples)

    def stop_recording(self, gh):
        cancel_future = gh.cancel_goal_async()
        rclpy.spin_until_future_complete(self, cancel_future)

        cancel_response = cancel_future.result()
        if cancel_response:
            self.get_logger().info('Action successfully canceled.')
        else:
            self.get_logger().info('Failed to cancel the action.')
            exit(0)

    def execute_movements(self):
        #### Setup and start recording
        gh = self.start_recording()
        print("Got goal:", gh)

        print(f"waiting for {self.min_samples} samples")
        while np.any(self.n_samples<self.min_samples): 
            time.sleep(0.05)
            rclpy.spin_once(self)
        print(f"got at least {self.min_samples} samples each: {self.n_samples}")

        #### Get grasping and wrist poses
        cloud_req = CloudPose.Request()
        cloud_req.offset_interval = self.offset_interval
        cloud_pose_res = call_cli_sync(self, self.pose_cli, cloud_req)

        offset = cloud_pose_res.offset
        grasp_pose = cloud_pose_res.grasp_pose
        grasp_wrist_pose = cloud_pose_res.wrist_pose
       
        #### Get varied grasping pose
        
        current_wrist_pose = transform_to_pose_stamped(self.tf_buffer.lookup_transform('map', 'wrist_3_link', rclpy.time.Time()))
        vary_req = CloudPoseVary.Request()
        vary_req.sampling_radius = self.sampling_radius
        vary_req.grasp_pose = grasp_pose
        vary_req.wrist_pose = current_wrist_pose

        vary_res = call_cli_sync(self, self.vary_pose_cli, vary_req)
        phi = vary_res.phi
        theta = vary_res.theta
        new_wrist_pose = vary_res.new_grasp_pose

        pa = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = new_wrist_pose.header.frame_id
        pa.poses = [
            grasp_pose.pose,
            grasp_wrist_pose.pose,
            current_wrist_pose.pose,
            new_wrist_pose.pose
        ]
        self.pose_array_publisher.publish(pa)
        
        ####
        #### Motion
        ####

        self.get_logger().info("opening gripper")
        gr = GripperService.Request()
        gr.open = True
        call_cli_sync(self, self.gripper_cli, gr)

        # self.get_logger().info("Moving to new gripper pose ...")
        # mr = MoveArm.Request()
        # mr.execute = True
        # mr.target_pose = new_wrist_pose
        # vary_pose_res = call_cli_sync(self, self.move_cli, mr)
        # print(vary_pose_res)

        self.get_logger().info("Moving to grasp pose...")
        mr = MoveArm.Request()
        mr.execute = True
        mr.target_pose = grasp_wrist_pose
        approach_pose_res = call_cli_sync(self, self.move_cli, mr)
        print(approach_pose_res)
        
        self.get_logger().info("inserting ...")
        pinsert = empty_pose(frame="wrist_3_link")
        pinsert.pose.position.z = 0.04

        mr = MoveArm.Request()
        mr.execute = True
        mr.target_pose = pinsert
        insert_pose_res = call_cli_sync(self, self.move_cli, mr)
        print(insert_pose_res)

        self.get_logger().info("closing gripper")
        gripper_close_time = datetime.now().timestamp()
        self.get_logger().info(f"{gripper_close_time}")
        gr = GripperService.Request()
        gr.open = False
        call_cli_sync(self, self.gripper_cli, gr)

        self.get_logger().info("lifting")
        plift = empty_pose(frame="wrist_3_link")
        plift.pose.position.x = 0.04

        mr = MoveArm.Request()
        mr.execute = True
        mr.target_pose = plift
        mr.execution_time = 0.5
        lift_pose_res = call_cli_sync(self, self.move_cli, mr)
        print(lift_pose_res)

        self.get_logger().info("retreating")
        pretr = empty_pose(frame="wrist_3_link")
        pretr.pose.position.z = -0.1

        mr = MoveArm.Request()
        mr.execute = True
        mr.target_pose = pretr
        mr.execution_time = 1.5
        retr_pose_res = call_cli_sync(self, self.move_cli, mr)
        print(retr_pose_res)

        self.get_logger().info("moving back to initial pose")
        mr = MoveArm.Request()
        mr.execute = True
        mr.name_target = approach_pose_res.name_start
        mr.q_target = approach_pose_res.q_start
        mr.execution_time = 2.0
        call_cli_sync(self, self.move_cli, mr)

        print("all done!")

        #### Store data
        self.stop_recording(gh)
        should_save = input("save? [Y/n]")

        if should_save.strip().lower() == "n":
            print("not saving data. bye.")
            return

        print("saving data ...")
        
        sample_dir = self.store_dir + "/" + datetime.now().strftime("%Y.%m.%d_%H.%M.%S") + "/"
        os.makedirs(sample_dir)

        with open(sample_dir+"misc.json", "w") as f:
            json.dump({
                "offset_interval": self.offset_interval,
                "offset": offset,
                "sampling_radius": self.sampling_radius,
                "phi": phi,
                "theta": theta,
                "grasp_pose": pose_to_list(grasp_pose),
                "grasp_wrist_pose": pose_to_list(grasp_wrist_pose),
                "new_wrist_pose": pose_to_list(new_wrist_pose),
                "current_wrist_pose": pose_to_list(current_wrist_pose),
                "gripper_close_time": gripper_close_time
            }, f)

        print("store request")
        store_req = StoreData.Request()
        store_req.dir = sample_dir
        call_cli_sync(self, self.store_cli, store_req)

        print("done!")


        

def main(args=None):
    rclpy.init(args=args)

    node = DataCollectionActionClient()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        node.execute_movements()
    except KeyboardInterrupt:
        print("Keyboard Interrupt!")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
