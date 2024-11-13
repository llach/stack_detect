import rclpy
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import PoseStamped
from tf2_geometry_msgs import PoseStamped

from stack_msgs.srv import CloudPose
from stack_msgs.action import RecordCloud

class CloudCollector(Node):
    """Subscriber node"""

    def __init__(self, executor=None):
        self.exe = executor

    def plan_timer(self):
        if self.pwrist is None:
            print("no desired wrist pose yet, waiting ...")
            return
        if self.current_q is None:
            print("no joint state yet, waiting ...")
            return
    
        self.pl.acquire()
        start_q = self.current_q.copy()
        
        print("approaching ...")
        pw = self.get_wrist_pose()
        pw.pose.position = self.pwrist.pose.position
        pw.pose.position.x += 0.01
        pw.pose.position.z -= 0.04

        approach_q = self.moveit_IK(start_q, self.pwrist)
        self.send_traj_blocking(approach_q, 3)

        print("inserting ...")
        pinsert = self.get_wrist_pose()
        pinsert.pose.position.z = 0.04

        insert_q = self.moveit_IK(approach_q, pinsert)
        self.send_traj_blocking(insert_q, 1)

        print("closing gripper")
        self.gripper.move_and_wait_for_pos(245, 0, 200)
        time.sleep(0.5)

        print("lifting")
        plift = self.get_wrist_pose()
        plift.pose.position.x = 0.04

        lift_q = self.moveit_IK(insert_q, plift)
        self.send_traj_blocking(lift_q, 1)

        print("retreating")
        pretr = self.get_wrist_pose()
        pretr.pose.position.z = -0.1

        retr_q = self.moveit_IK(lift_q, pretr)
        self.send_traj_blocking(retr_q, 1.5)

        print("moving back to initial pose")
        self.send_traj_blocking(start_q, 1.5)

        print("all done!")
        self.destroy_node()
        self.exe.shutdown()

    def empty_wrist_pose(self):
        p = PoseStamped()
        p.header.stamp = self.get_clock().now().to_msg()
        p.header.frame_id = "wrist_3_link"
        p.pose.position.x = 0.0
        p.pose.position.y = 0.0
        p.pose.position.z = 0.0
        p.pose.orientation.x = 0.0
        p.pose.orientation.y = 0.0
        p.pose.orientation.z = 0.0
        p.pose.orientation.w = 1.0
        return p

class DataCollectionActionClient(Node):
    def __init__(self, min_samples):
        super().__init__('data_collection_action_client')
        self._action_client = ActionClient(self, RecordCloud, 'collect_cloud_data', callback_group=ReentrantCallbackGroup())
        self.min_samples = min_samples
        self.executing = False

        self.pose_cli = self.create_client(CloudPose, 'get_cloud_pose')
        while not self.pose_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('pose service not available, waiting again...')

    def send_goal(self):
        # Create a goal to start data collection
        goal_msg = RecordCloud.Goal()
        goal_msg.start = True  # Set the start flag

        # Wait for the action server to be available
        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()

        # Send the goal
        self.get_logger().info('Sending goal to start data collection...')
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        # Handle the response to the goal
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected by action server.')
            return

        self.get_logger().info('Goal accepted by action server.')
        self._goal_handle = goal_handle

        # Monitor result of the goal asynchronously
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        # Receive feedback from the action server
        joint_state_samples = feedback_msg.feedback.joint_state_samples
        transform_samples = feedback_msg.feedback.transform_samples

        # Print feedback
        # self.get_logger().info(f'Feedback: Joint State Samples: {joint_state_samples}, Transform Samples: {transform_samples}')

        # Check if we've collected enough samples
        if joint_state_samples >= self.min_samples and transform_samples >= self.min_samples:
            # Prompt user input once min_samples is reached
            if not self.executing: self.execute_movements()

    def cancel_goal(self):
        # Cancel the goal on the action server
        self.get_logger().info('Requesting cancellation of the action...')
        cancel_future = self._goal_handle.cancel_goal_async()
        cancel_future.add_done_callback(self.cancel_done_callback)

    def cancel_done_callback(self, future):
        cancel_response = future.result()
        if cancel_response:
            self.get_logger().info('Action successfully canceled.')
        else:
            self.get_logger().info('Failed to cancel the action.')

    def get_result_callback(self, future):
        result = future.result().result
        if result.success:
            self.get_logger().info('Data collection action completed successfully.')
        else:
            self.get_logger().info('Data collection action did not complete successfully.')

    def execute_movements(self):
        if self.executing: return

        self.get_logger().info(f'Collected {self.min_samples} samples for both joint states and transforms. Moveing the robot.')
        self.executing = True
        

        for i in range(10):
            print("calling service .........")
            fut = self.pose_cli.call_async(CloudPose.Request())
            rclpy.spin_until_future_complete(self, fut)
            print(fut.result())

        should_save = input("save? [Y/n]")

        # Cancel the goal after user input
        self.cancel_goal()


def main(args=None):
    rclpy.init(args=args)
    min_samples = 10  # Set the minimum number of samples required

    action_client = DataCollectionActionClient(min_samples)
    action_client.send_goal()

    rclpy.spin(action_client)

    action_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
