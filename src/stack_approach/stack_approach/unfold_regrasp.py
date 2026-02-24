#!/usr/bin/env python3
import cv2
import time
import rclpy
import numpy as np



from motion_helper_v2 import MotionHelperV2

import numpy as np
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PoseStamped, Quaternion


"""

############################################################ DINOV3 Touch primitive final pose

dist y = |-0.110| + 0.040 = 0.150

map -> left_arm_wrist_3_link

[INFO] [1770970995.505082068] [tf2_echo]: Waiting for transform map ->  left_arm_wrist_3_link: Invalid frame ID "map" passed to canTransform argument target_frame - frame does not exist
At time 1770970996.456426888
- Translation: [0.365, -0.110, 0.825]
- Rotation: in Quaternion (xyzw) [-0.100, 0.981, 0.083, 0.142]
- Rotation: in RPY (radian) [3.000, 0.300, -2.960]
- Rotation: in RPY (degree) [171.886, 17.188, -169.588]
- Matrix:
 -0.940 -0.220  0.262  0.365
 -0.173  0.966  0.192 -0.110
 -0.296  0.135 -0.946  0.825
  0.000  0.000  0.000  1.000

  
map -> right_arm_wrist_3_link

[INFO] [1770971031.069038476] [tf2_echo]: Waiting for transform map ->  right_arm_wrist_3_link: Invalid frame ID "map" passed to canTransform argument target_frame - frame does not exist
At time 1770971032.36428275
- Translation: [0.381, 0.040, 0.817]
- Rotation: in Quaternion (xyzw) [0.396, 0.908, -0.009, 0.132]
- Rotation: in RPY (radian) [3.050, 0.250, 2.307]
- Rotation: in RPY (degree) [174.752, 14.323, 132.197]
- Matrix:
 -0.651  0.723  0.233  0.381
  0.718  0.686 -0.121  0.040
 -0.247  0.089 -0.965  0.817
  0.000  0.000  0.000  1.000

  # Current joint positions for: both [rad]
[
    0.9284,
    -1.9090,
    -1.8429,
    -1.8748,
    -1.0625,
    2.3814,
    -0.8663,
    -1.0480,
    1.7283,
    -1.4043,
    1.1222,
    1.3825,
]


# Current joint positions for: both [deg]
[
    53.19,
    -109.38,
    -105.59,
    -107.42,
    -60.88,
    136.44,
    -49.63,
    -60.05,
    99.02,
    -80.46,
    64.29,
    79.21,
]

############################################################ YOLO start pose

dist y = |-0.089| + 0.089 = 0.178

[INFO] [1770971324.813461649] [tf2_echo]: Waiting for transform map ->  left_arm_wrist_3_link: Invalid frame ID "map" passed to canTransform argument target_frame - frame does not exist
At time 1770971325.770426867
- Translation: [0.603, -0.089, 1.191]
- Rotation: in Quaternion (xyzw) [0.663, 0.748, -0.000, 0.000]
- Rotation: in RPY (radian) [-3.142, 0.000, 1.692]
- Rotation: in RPY (degree) [-180.000, 0.002, 96.916]
- Matrix:
 -0.120  0.993 -0.000  0.603
  0.993  0.120 -0.000 -0.089
 -0.000 -0.000 -1.000  1.191
  0.000  0.000  0.000  1.000

[INFO] [1770971329.096028718] [tf2_echo]: Waiting for transform map ->  right_arm_wrist_3_link: Invalid frame ID "map" passed to canTransform argument target_frame - frame does not exist
At time 1770971330.74433064
- Translation: [0.603, 0.089, 1.191]
- Rotation: in Quaternion (xyzw) [-0.406, 0.914, -0.000, -0.000]
- Rotation: in RPY (radian) [-3.141, -0.000, -2.306]
- Rotation: in RPY (degree) [-179.992, -0.017, -132.105]
- Matrix:
 -0.670 -0.742 -0.000  0.603
 -0.742  0.670 -0.000  0.089
  0.000 -0.000 -1.000  1.191
  0.000  0.000  0.000  1.000

# Current joint positions for: both [rad]
[
    1.4855,
    -2.6960,
    -1.6677,
    -0.4337,
    -0.7888,
    3.1419,
    -1.4855,
    -0.4457,
    1.6677,
    -2.7078,
    0.7888,
    0.6136,
]


# Current joint positions for: both [deg]
[
    85.11,
    -154.47,
    -95.55,
    -24.85,
    -45.20,
    180.02,
    -85.11,
    -25.53,
    95.55,
    -155.15,
    45.20,
    35.16,
]
"""

DINO_TRANS_LEFT = [0.365, -0.110, 0.825]
DINO_TRANS_RIGHT = [0.381, 0.040, 0.817]

DINO_ROT_LEFT = [-0.100, 0.981, 0.083, 0.142]
DINO_ROT_RIGHT = [0.396, 0.908, -0.009, 0.132]

YOLO_TRANS_LEFT = [0.603, -0.089, 1.191]
YOLO_TRANS_RIGHT = [0.603, 0.089, 1.191]

YOLO_ROT_LEFT = [0.663, 0.748, -0.000, 0.000]
YOLO_ROT_RIGHT = [-0.406, 0.914, -0.000, -0.000]


DINOV3_END_Q = [
    0.9284,
    -1.9090,
    -1.8429,
    -1.8748,
    -1.0625,
    2.3814,
    -0.8663,
    -1.0480,
    1.7283,
    -1.4043,
    1.1222,
    1.3825,
]

YOLO_START_Q = [
    1.4855,
    -2.6960,
    -1.6677,
    -0.4337,
    -0.7888,
    3.1419,
    -1.4855,
    -0.4457,
    1.6677,
    -2.7078,
    0.7888,
    0.6136,
]

LEFT_TRANS = [
    0.9616,
    -1.8451,
    -1.8693,
    -1.9008,
    -1.0364,
    2.4538,
]

LEFT_ROT = [
    0.9616,
    -1.8451,
    -1.8693,
    -1.9008,
    -1.0364,
    3.2646,
]

RIGHT_TRANS = [
    -0.8900,
    -1.1016,
    1.7613,
    -1.3917,
    1.1042,
    1.4001,
]

RIGHT_ROT = [
    -0.8900,
    -1.1016,
    1.7613,
    -1.3916,
    1.1042,
    0.5489,
]


from stack_approach.helpers import transform_to_pose_stamped
from stack_msgs.srv import MoveArm, RollerGripper, RollerGripperV2

if __name__ == '__main__':
    rclpy.init()

    prim_time = 6
    mh2 = MotionHelperV2()

    mh2.call_cli_sync(mh2.finger2srv["left_v2"], RollerGripperV2.Request(position=1.0))
    mh2.call_cli_sync(mh2.finger2srv["right_v2"], RollerGripperV2.Request(position=1.0))

    input("inp")
    mh2.go_to_q(
        q = DINOV3_END_Q,
        time = 5,
        side = "both"
    )

    input("inp")
    mh2.call_cli_sync(mh2.finger2srv["right_v2"], RollerGripperV2.Request(position=-1.0))
    time.sleep(0.5)

    mh2.go_to_q(
        q = LEFT_TRANS,
        time = prim_time,
        side = "left"
    )

    mh2.go_to_q(
        q = LEFT_ROT,
        time = prim_time,
        side = "left"
    )

    mh2.call_cli_sync(mh2.finger2srv["left_v2"], RollerGripperV2.Request(position=-1.0))
    time.sleep(0.2)
    mh2.call_cli_sync(mh2.finger2srv["right_v2"], RollerGripperV2.Request(position=1.0))
    time.sleep(0.5)

    mh2.go_to_q(
        q=RIGHT_TRANS,
        time = prim_time,
        side="right"
    )

    mh2.go_to_q(
        q=RIGHT_ROT,
        time = prim_time,
        side="right"
    )

    mh2.call_cli_sync(mh2.finger2srv["right_v2"], RollerGripperV2.Request(position=-1.0))
    time.sleep(0.7)

    input("yolo")
    mh2.go_to_q(
        q = YOLO_START_Q,
        time = 10,
        side = "both"
    )



    # pub = mh2.create_publisher(PoseStamped, '/unstack_start_pose', 10)

    # time.sleep(2)
    # print("LETS GOOOOOO")

    # msg = transform_to_pose_stamped(mh2.tf_buffer.lookup_transform("map", f"right_arm_wrist_3_link", rclpy.time.Time()))

    # print(msg)

    # msg.pose.orientation = Quaternion(
    #     x=YOLO_ROT_RIGHT[0],
    #     y=YOLO_ROT_RIGHT[1],
    #     z=YOLO_ROT_RIGHT[2],
    #     w=YOLO_ROT_RIGHT[3],
    # )

    # q = mh2.compute_ik_with_retries(msg, mh2.current_q.copy(), side="right")
    # print(q)

    # exit(0)

    # mh2.go_to_q(
    #     q = DINOV3_END_Q,
    #     time = 5,
    #     side = "both"
    # )

    # input("inp")
    # mh2.call_cli_sync(mh2.finger2srv["right_v2"], RollerGripperV2.Request(position=-1.0))

    # input("inp")
    # mh2.call_cli_sync(mh2.finger2srv["left_v2"], RollerGripperV2.Request(position=0.2))



    rclpy.shutdown()
