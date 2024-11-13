from stack_msgs.srv import CloudPoseVary

import rclpy
import numpy as np
from rclpy.node import Node
import tf_transformations as tf
from geometry_msgs.msg import PoseStamped, Pose



class CloudPoseVaryService(Node):

    def __init__(self):
        super().__init__('CloudPoseVary')

        self.srv = self.create_service(CloudPoseVary, 'cloud_pose_vary', self.srv_callback)

    def srv_callback(self, req, res):
        center = np.array([
            req.grasp_pose.pose.position.x,
            req.grasp_pose.pose.position.y,
            req.grasp_pose.pose.position.z
        ])

        wrist_position = np.array([
            req.wrist_pose.pose.position.x,
            req.wrist_pose.pose.position.y,
            req.wrist_pose.pose.position.z
        ])
        wrist_orientation = np.array([
            req.wrist_pose.pose.orientation.x,
            req.wrist_pose.pose.orientation.y,
            req.wrist_pose.pose.orientation.z,
            req.wrist_pose.pose.orientation.w
        ])
        radius = np.linalg.norm(wrist_position - center)


        # Calculate latitude (phi) limit for the sampling radius
        max_phi = np.arccos((radius - req.sampling_radius) / radius) if req.sampling_radius < radius else np.pi
        
        # Generate spherical cap points around the Z-axis, which we will later rotate
        phi = np.random.uniform(0, max_phi)  # Restricted latitude within cap
        theta = np.random.uniform(0, 2 * np.pi)  # Full longitude

        res.phi = phi
        res.theta = theta
        res.new_grasp_pose = self.generate_sphere_pose(
            center=center,
            radius=radius,
            phi=phi,
            theta=theta,
            wrist_position=wrist_position,
            wrist_orientation=wrist_orientation,
            center_pose=req.grasp_pose
        )

        print(res.new_grasp_pose)
        return res
    
    def generate_sphere_pose(self, center, radius, phi, theta, wrist_position, wrist_orientation, center_pose):
        # Calculate the direction vector from the center to the initial wrist_3_pose
        wrist_direction = wrist_position - center
        wrist_direction /= np.linalg.norm(wrist_direction)  # Normalize
        
        # Generate the point on the sphere in the local Z-axis direction
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        
        # Rotate this point to align with the wrist_3_pose direction
        point = np.array([x, y, z])
        rotation_axis = np.cross([0, 0, 1], wrist_direction)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        if rotation_axis_norm > 1e-6:
            rotation_axis /= rotation_axis_norm
            rotation_angle = np.arccos(np.dot([0, 0, 1], wrist_direction))
            rotation_quaternion = tf.quaternion_about_axis(rotation_angle, rotation_axis)
            rotated_point = tf.quaternion_matrix(rotation_quaternion)[:3, :3].dot(point)
        else:
            # No rotation needed if already aligned
            rotated_point = point

        # Translate rotated point to be centered around the original center
        x, y, z = rotated_point + center
        
        # Create Pose
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z

        q = wrist_orientation
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]

        ps = PoseStamped()
        ps.pose = pose

        return self.align_local_z_axis_to_target(ps, center_pose)


    def align_local_z_axis_to_target(self, frame1, frame2):
        """
        Rotates frame1 such that its local Z-axis points toward the center of frame2,
        while preserving frame1's original X/Y orientation.
        
        :param frame1: PoseStamped representing the first frame.
        :param frame2: PoseStamped representing the second frame.
        :return: PoseStamped with modified orientation for frame1.
        """
        # Step 1: Calculate the direction vector from frame1 to frame2 in the parent frame
        direction_parent = np.array([
            frame2.pose.position.x - frame1.pose.position.x,
            frame2.pose.position.y - frame1.pose.position.y,
            frame2.pose.position.z - frame1.pose.position.z
        ])
        direction_parent /= np.linalg.norm(direction_parent)  # Normalize the direction vector

        # Step 2: Rotate this direction into frame1's local coordinate system
        # Convert frame1's orientation quaternion to a rotation matrix
        original_orientation = [
            frame1.pose.orientation.x,
            frame1.pose.orientation.y,
            frame1.pose.orientation.z,
            frame1.pose.orientation.w
        ]
        rotation_matrix = tf.quaternion_matrix(original_orientation)[:3, :3]
        
        # Transform the direction vector into frame1's local space
        local_direction = rotation_matrix.T.dot(direction_parent)

        # Step 3: Calculate the rotation needed to align frame1's local Z-axis with this local direction
        z_axis_local = np.array([0, 0, 1])  # Z-axis in frame1's local space
        rotation_axis = np.cross(z_axis_local, local_direction)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        if rotation_axis_norm > 1e-6:  # If rotation is required
            rotation_axis /= rotation_axis_norm  # Normalize the rotation axis
            angle = np.arccos(np.dot(z_axis_local, local_direction))  # Calculate the rotation angle
            alignment_quaternion = tf.quaternion_about_axis(angle, rotation_axis)
        else:
            # If already aligned or exactly opposite
            if np.dot(z_axis_local, local_direction) < 0:
                # Opposite, rotate 180 degrees around any perpendicular axis
                alignment_quaternion = tf.quaternion_about_axis(np.pi, [1, 0, 0])
            else:
                alignment_quaternion = [0.0, 0.0, 0.0, 1.0]  # Already aligned

        # Step 4: Combine the alignment quaternion with the original orientation of frame1
        new_orientation = tf.quaternion_multiply(original_orientation, alignment_quaternion)

        # Step 5: Update frame1's orientation
        aligned_frame = PoseStamped()
        aligned_frame.header = frame2.header
        aligned_frame.pose.position = frame1.pose.position  # Keep the original position
        aligned_frame.pose.orientation.x = new_orientation[0]
        aligned_frame.pose.orientation.y = new_orientation[1]
        aligned_frame.pose.orientation.z = new_orientation[2]
        aligned_frame.pose.orientation.w = new_orientation[3]

        return aligned_frame

def main():
    rclpy.init()

    minimal_service = CloudPoseVaryService()

    rclpy.spin(minimal_service)
    rclpy.shutdown()


if __name__ == '__main__':
    main()