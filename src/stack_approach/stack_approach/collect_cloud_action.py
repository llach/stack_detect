import os
import json
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from tf2_ros import TransformListener, Buffer
from stack_msgs.action import RecordCloud  # Import the renamed action
import threading
from datetime import datetime

class DataCollectionActionServer(Node):
    def __init__(self, store_dir = f"{os.environ['HOME']}/unstack_cloud"):
        super().__init__('data_collection_action_server')
        self.store_dir = store_dir

        # Action server setup
        self._action_server = ActionServer(
            self,
            RecordCloud,  # Use the renamed action
            'collect_cloud_data',  # Action name
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            handle_accepted_callback=self.handle_accepted_callback,
            cancel_callback=self.cancel_callback
        )

        # Transform listener setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Storage for joint states and transforms
        self.joint_states = []
        self.transforms = []

        # Subscription to joint states
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Control flags for data collection
        self.collecting_data = False
        print("accepting goals!")

    def goal_callback(self, goal_request):
        # Accept all incoming goals
        self.get_logger().info("Received goal request for data collection.")
        return GoalResponse.ACCEPT

    def handle_accepted_callback(self, goal_handle):
        # Execute the goal in a new coroutine to allow for concurrency
        goal_handle.execute()

    def cancel_callback(self, goal_handle):
        self.get_logger().info("Cancelling data collection...")

        self.collecting_data = False
        print(f"Nsamples:\njs: {len(self.joint_states)} | tf: {len(self.transforms)}")

        sample_dir = self.store_dir + "/" + datetime.now().strftime("%Y.%m.%d_%H.%M.%S") + "/"
        os.makedirs(sample_dir)

        with open(sample_dir+"joint_states.json", "w") as f:
            json.dump(self.joint_states, f)

        with open(sample_dir+"gripper_poses.json", "w") as f:
            json.dump(self.transforms, f)

        print("done saving.")

        self.clear_buffers()
        return CancelResponse.ACCEPT

    def joint_state_callback(self, msg):
        if self.collecting_data:
            # Store joint positions by name
            self.joint_states.append({
                "timestamp": int(datetime.now().timestamp()),
                "joint_states": dict(map(lambda i,j: (i,j),  msg.name, msg.position))
            })

    def get_transform_data(self):
        try:
            transform = self.tf_buffer.lookup_transform('map', 'wrist_3_link', rclpy.time.Time())
            timestamp = int(datetime.now().timestamp())
            position = [transform.transform.translation.x,
                        transform.transform.translation.y,
                        transform.transform.translation.z]
            orientation = [transform.transform.rotation.x,
                           transform.transform.rotation.y,
                           transform.transform.rotation.z,
                           transform.transform.rotation.w]

            return {'timestamp': timestamp, 'position': position, 'orientation': orientation}
        except Exception as e:
            self.get_logger().warn(f"Transform not found: {e}")
            return None

    def execute_callback(self, goal_handle):
        self.get_logger().info('Starting data collection action...')

        # Initialize data collection
        self.collecting_data = True

        # Loop until the goal is canceled
        r = self.create_rate(10)
        while rclpy.ok() and self.collecting_data:
            transform_data = self.get_transform_data()
            if transform_data:
                self.transforms.append(transform_data)

            # Send feedback with the count of collected samples
            feedback_msg = RecordCloud.Feedback()
            feedback_msg.joint_state_samples = len(self.joint_states)
            feedback_msg.transform_samples = len(self.transforms)
            goal_handle.publish_feedback(feedback_msg)

            # Allow time for new joint state messages
            r.sleep()

        # Stop data collection and complete the action
        self.collecting_data = False
        self.get_logger().info('Data collection canceled.')

        # Clear data buffers after completion
        self.clear_buffers()

        result = RecordCloud.Result()
        result.success = True
        goal_handle.succeed()
        return result

    def clear_buffers(self):
        """Clear the data buffers for joint states and transforms."""
        self.joint_states = []
        self.transforms = []
        self.get_logger().info("Data buffers cleared.")

def main(args=None):
    rclpy.init(args=args)
    node = DataCollectionActionServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
