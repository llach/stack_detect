"""
tui_bagopen_utils.py
====================
Self-contained bag opening logic for use by tui_controller.py.

Clean copy of the relevant parts of bag_opening_perc.py with every
active input() call replaced by _ask(), which either:

  - shows a TUI modal and waits for Y/N + Enter  (all prompts here need it)
  - auto-confirms silently                        (commented-out ones, none active)

bag_opening_perc.py is NOT imported and NOT needed at runtime.
tui_controller.py imports only this file.

Public API
----------
  run(state, bag_node, mh2)
      Called from a daemon thread in tui_controller. Runs the full bag
      opening sequence (equivalent to bag_opening_perc.main() with
      with_slides=True).

  TrajectoryPublisher
      The ROS node class. Instantiated once in tui_controller.main() and
      passed to run().

input() routing
---------------
  "################### move? (y/N)"  →  TUI modal, y continues / anything else skips
  "init pose? (Y/n)"                 →  TUI modal, anything != "n" runs initial pose

  Both use the same UnstackGate mechanism as tui_unstack_utils.
"""

import os
import glob
import time
import rclpy
import numpy as np

from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from control_msgs.action import FollowJointTrajectory
from softenable_display_msgs.srv import SetDisplay
from stack_msgs.srv import RollerGripper, StackDetect, MoveArm, RollerGripperV2, RollerGripperV3
from stack_approach.motion_helper_v2 import MotionHelperV2
from stack_approach.controller_switcher import ControllerSwitcher
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Point, WrenchStamped
from stack_approach.helpers import empty_pose
from std_msgs.msg import Bool


# ═══════════════════════════════════════════════════════════════════════════
#  input() routing  (same pattern as tui_unstack_utils)
# ═══════════════════════════════════════════════════════════════════════════

# All active prompts in this file require confirmation — none are auto-confirmed.
# _ask_fn is replaced by run() for the duration of the sequence.
_ask_fn = input


def _ask(prompt: str = "") -> str:
    return _ask_fn(prompt)


# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

class BagType:
    NORMAL = "normal"
    THIN   = "thin"

move_groups = ["both", "both", "left", "left", "right", "right", "both", "both", "both"]

group2joints = {
    "both": [
        "left_arm_shoulder_pan_joint",  "left_arm_shoulder_lift_joint",
        "left_arm_elbow_joint",         "left_arm_wrist_1_joint",
        "left_arm_wrist_2_joint",       "left_arm_wrist_3_joint",
        "right_arm_shoulder_pan_joint", "right_arm_shoulder_lift_joint",
        "right_arm_elbow_joint",        "right_arm_wrist_1_joint",
        "right_arm_wrist_2_joint",      "right_arm_wrist_3_joint",
    ],
    "left": [
        "left_arm_shoulder_pan_joint", "left_arm_shoulder_lift_joint",
        "left_arm_elbow_joint",        "left_arm_wrist_1_joint",
        "left_arm_wrist_2_joint",      "left_arm_wrist_3_joint",
    ],
    "right": [
        "right_arm_shoulder_pan_joint", "right_arm_shoulder_lift_joint",
        "right_arm_elbow_joint",        "right_arm_wrist_1_joint",
        "right_arm_wrist_2_joint",      "right_arm_wrist_3_joint",
    ],
}

group2controller = {
    "both":  "dual_arm_joint_trajectory_controller",
    "left":  "left_arm_joint_trajectory_controller",
    "right": "right_arm_joint_trajectory_controller",
}

PRE_PLACE_LEFT = [2.2566, -1.6983, -2.0718, 0.8025, 0.5113, -3.2023]
PLACE_LEFT     = [2.3772, -1.2721, -1.8276, 1.4037, 0.9858, -4.5638]

START_POSE_SHIFTED_LEFT_STUDY = [
     2.3272, -0.8861, -2.2243, -0.9957, -1.0295,  0.8760,
    -0.9561, -0.1050,  1.4287, -2.3651,  0.9607, -0.5481,
]

TRAJECTORIES_DIR = "/home/ros/ws/src/bag_opening/trajectories/"


def adjust_ts(ts, scaling=1, offset=0):
    return ((ts - ts[0]) * scaling) + offset


# ═══════════════════════════════════════════════════════════════════════════
#  TrajectoryPublisher node  (verbatim from bag_opening_perc.py)
# ═══════════════════════════════════════════════════════════════════════════

class TrajectoryPublisher(Node):
    def __init__(self, with_slides: bool = False):
        super().__init__('bag_opening_perc')

        self.with_slides = with_slides
        self.recbg = ReentrantCallbackGroup()
        self.controller_switcher = ControllerSwitcher()

        self.group2client = {
            "both": ActionClient(
                self, FollowJointTrajectory,
                "/dual_arm_joint_trajectory_controller/follow_joint_trajectory",
                callback_group=self.recbg),
            "left": ActionClient(
                self, FollowJointTrajectory,
                "/left_arm_joint_trajectory_controller/follow_joint_trajectory",
                callback_group=self.recbg),
            "right": ActionClient(
                self, FollowJointTrajectory,
                "/right_arm_joint_trajectory_controller/follow_joint_trajectory",
                callback_group=self.recbg),
        }

        self.finger2srv = {
            "left":    self.create_client(RollerGripper,   'left_roller_gripper'),
            "right":   self.create_client(RollerGripper,   'right_roller_gripper'),
            "left_v2": self.create_client(RollerGripperV2, 'left_gripper_normalized'),
            "right_v2":self.create_client(RollerGripperV2, 'right_gripper_normalized'),
            "left_v3": self.create_client(RollerGripperV3, 'left_gripper_effort'),
            "right_v3":self.create_client(RollerGripperV3, 'right_gripper_effort'),
        }

        for n, srv in self.finger2srv.items():
            print("waiting for", n)
            srv.wait_for_service(timeout_sec=1.0)

        self.latest_ft      = None
        self.latest_ft_time = None
        self.data_timeout   = 0.1

        self.create_subscription(WrenchStamped, '/ur5/ft_raw', self.ft_callback, 10)

        for k, v in self.finger2srv.items():
            print(f"waiting for {k.upper()} gripper srv")
            while not v.wait_for_service(timeout_sec=2.0):
                self.get_logger().info(f'{k} service not available, waiting again...')
            print(f"found {k.upper()} gripper srv")

        if with_slides:
            self.cli_display = self.create_client(SetDisplay, '/set_display')
            while not self.cli_display.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('Waiting for /set_display service...')

        self.move_cli = self.create_client(MoveArm, "move_arm", callback_group=self.recbg)
        while not self.move_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('move arm service not available, waiting again...')

        self.bag_cli = self.create_client(StackDetect, "detect_bag", callback_group=self.recbg)
        while not self.bag_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('bag detect service not available, waiting again...')

    def switch_slide(self, slide_name):
        if not self.with_slides:
            return
        self.cli_display.call_async(SetDisplay.Request(name=slide_name))

    def has_fresh_ft(self):
        if self.latest_ft is None or self.latest_ft_time is None:
            return False
        return (time.time() - self.latest_ft_time) < self.data_timeout

    def ft_callback(self, msg: WrenchStamped):
        f = msg.wrench.force
        self.latest_ft      = [f.x, f.y, f.z]
        self.latest_ft_time = time.time()

    def wait_for_force_change(self, threshold: float, sleep_time: float = 0.05):
        while rclpy.ok() and not self.has_fresh_ft():
            rclpy.spin_once(self, timeout_sec=0.1)
        f_ref = self.latest_ft.copy()
        if f_ref is None:
            self.get_logger().error("No F/T data available for reference")
            return False
        self.get_logger().info(f"Starting force monitor, ref={f_ref}")
        start = time.time()
        while rclpy.ok() and (time.time() - start) < 180:
            rclpy.spin_once(self, timeout_sec=0.01)
            f_cur = self.latest_ft.copy()
            if f_cur is None:
                continue
            diff_sum = sum(abs(fc - fr) for fc, fr in zip(f_cur, f_ref))
            if diff_sum >= threshold:
                self.get_logger().info(
                    f"Force threshold exceeded: sum(|ΔF|)={diff_sum:.2f} >= {threshold}")
                return True
            self.ros_sleep(sleep_time)
        print("FT release timeout!")
        return False

    def ros_sleep(self, sec):
        for _ in range(int(sec / 0.1)):
            time.sleep(0.1)
            rclpy.spin_once(self)

    def execute_traj(self, group, ts, qs):
        assert group in ["both", "left", "right"], f"unknown move group: {group}"
        print(f"executing trajectory with {len(qs)} points in {ts[-1]:.2f}s ...")
        self.controller_switcher.activate_controller(group2controller[group])
        traj = JointTrajectory()
        traj.points      = []
        traj.joint_names = group2joints[group]
        for wp, t in zip(qs, ts):
            point = JointTrajectoryPoint()
            point.positions = list(wp)
            point.time_from_start.sec    = int(t)
            point.time_from_start.nanosec = int((t % 1.0) * 1e9)
            traj.points.append(point)
        traj_goal = FollowJointTrajectory.Goal()
        traj_goal.trajectory = traj
        self.get_logger().info('{} waypoints [{} -> {}]'.format(
            len(traj.points), ts[0], ts[-1]))
        return self.group2client[group].send_goal_async(traj_goal)

    def call_cli_sync(self, cli, req):
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        return fut.result()

    def initial_pose_new(self, dur=4):
        fut = self.execute_traj(
            "both", np.array([dur]), np.array([START_POSE_SHIFTED_LEFT_STUDY]))
        await_action_future(self, fut)


# ═══════════════════════════════════════════════════════════════════════════
#  Helper functions  (verbatim from bag_opening_perc.py)
# ═══════════════════════════════════════════════════════════════════════════

def load_trajectories(folder):
    files = glob.glob(os.path.join(folder, "t*.npz"))
    if not files:
        print("No files found.")
        return
    files = sorted(files, key=lambda f: int(os.path.splitext(os.path.basename(f))[0][1:]))
    print("Found files:", files)
    arrays = []
    for f in files:
        data = np.load(f)
        arrays.append(dict(q=data["q"], ts=data["timestamps"]))
    return arrays


def await_action_future(node, fut):
    print("waiting for future ...")
    while not fut.done() and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
    print("future done!")
    goal_handle = fut.result()
    if not goal_handle.accepted:
        print("Trajectory was rejected by the controller!")
        return False
    future_result = goal_handle.get_result_async()
    print("waiting for future result ...")
    while not future_result.done() and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
    print("future result done!")
    result = future_result.result()
    print(f"Trajectory finished with status: {result.status}")
    return True


def move_over_bag(node: TrajectoryPublisher, left_wrist_pose, execution_time=1.5):
    mar_left_pre = MoveArm.Request(
        target_pose     = left_wrist_pose,
        execute         = False,
        controller_name = "left_arm_joint_trajectory_controller",
        execution_time  = float(execution_time),
        ik_link         = "left_arm_wrist_3_link",
        name_target     = ["left_arm_shoulder_pan_joint", "left_arm_shoulder_lift_joint",
                           "left_arm_elbow_joint",        "left_arm_wrist_1_joint",
                           "left_arm_wrist_2_joint",      "left_arm_wrist_3_link"],
    )
    fut = node.move_cli.call_async(mar_left_pre)
    rclpy.spin_until_future_complete(node, fut)
    final_pose = np.concatenate([
        fut.result().q_end,
        np.deg2rad([-94.88, -51.86, 135.98, -179.93, 45.55, -72.64])
    ])
    print("over bag pose", final_pose)
    fut = node.execute_traj("both", np.array([3.0]), np.array([final_pose]))
    await_action_future(node, fut)


def execute_trajectories(node: TrajectoryPublisher, arrays):
    offsets    = [0 for _ in range(len(arrays))]
    offsets[0] = 10
    for i, arr in enumerate(arrays):
        if i < 4:
            continue
        print(f"--- T{i} ---")
        ts = adjust_ts(arr["ts"], offset=offsets[i], scaling=0.7 if i == 2 else 0.7)
        if i == 5:
            node.finger2srv["right"].call_async(
                RollerGripper.Request(roller_vel=-80, roller_duration=ts[-1] * 0.95))
        fut = node.execute_traj(move_groups[i], ts, arr["q"])
        if fut:
            if not await_action_future(node, fut):
                break
        if i == 5:
            node.call_cli_sync(node.finger2srv["right_v3"],
                               RollerGripperV3.Request(effort=0.3))
    node.execute_traj(
        "both", np.array([2.5]),
        np.array([np.deg2rad([
            74.16, -112.87, -94.81, -33.03,  89.34, -183.47,
           -68.02,  -47.74,  62.89, -324.02,  88.60,   32.82,
        ])]))


def execute_opening(mh2: MotionHelperV2, node: TrajectoryPublisher,
                    trajs, bag_type: BagType, with_slides: bool = False):
    PRE_GRASP_HEIGHT = 0.837
    X_OFFSET = 0.035
    if bag_type == BagType.NORMAL:
        GRASP_HEIGHT   = 0.7917
        TARGET_FORCE   = 20
        Y_OFFSET       = -0.04
        ROLL_TIME_LEFT = 4.5
    elif bag_type == BagType.THIN:
        GRASP_HEIGHT   = 0.792
        TARGET_FORCE   = 14
        Y_OFFSET       = 0.013
        ROLL_TIME_LEFT = 5.0

    res = None
    while True:
        print("calling bag_cli")
        fut = node.bag_cli.call_async(StackDetect.Request(
            offset=Point(x=X_OFFSET, y=Y_OFFSET, z=PRE_GRASP_HEIGHT)))
        rclpy.spin_until_future_complete(node, fut)
        res = fut.result()

        # ── Confirmation modal: operator checks the bag pose ───────────────
        inp = _ask("################### move? (y/N)").lower().strip()
        if inp == "q":
            return
        elif inp == "y":
            break

    if not res:
        print("ERROR res is None")
        return

    bag_pose_wrist = res.target_pose
    print(bag_pose_wrist)

    move_over_bag(node, bag_pose_wrist, execution_time=5)

    grasp_pose = empty_pose(frame="left_arm_wrist_3_link")
    grasp_pose.pose.position.z = PRE_GRASP_HEIGHT - GRASP_HEIGHT

    fut = node.move_cli.call_async(MoveArm.Request(
        execute         = True,
        target_pose     = grasp_pose,
        execution_time  = 1.7,
        controller_name = "left_arm_joint_trajectory_controller",
        ik_link         = "left_arm_wrist_3_link",
        name_target     = ["left_arm_shoulder_pan_joint", "left_arm_shoulder_lift_joint",
                           "left_arm_elbow_joint",        "left_arm_wrist_1_joint",
                           "left_arm_wrist_2_joint",      "left_arm_wrist_3_joint"],
    ))
    rclpy.spin_until_future_complete(node, fut)

    time.sleep(0.5)
    node.call_cli_sync(node.finger2srv["left"],
                       RollerGripper.Request(roller_vel=80, roller_duration=ROLL_TIME_LEFT))
    node.call_cli_sync(node.finger2srv["left_v3"], RollerGripperV3.Request(effort=0.3))

    grasp_pose_up = empty_pose(frame="left_arm_wrist_3_link")
    grasp_pose_up.pose.position.z = -0.02

    fut = node.move_cli.call_async(MoveArm.Request(
        execute         = True,
        target_pose     = grasp_pose_up,
        execution_time  = 0.5,
        controller_name = "left_arm_joint_trajectory_controller",
        ik_link         = "left_arm_wrist_3_link",
        name_target     = ["left_arm_shoulder_pan_joint", "left_arm_shoulder_lift_joint",
                           "left_arm_elbow_joint",        "left_arm_wrist_1_joint",
                           "left_arm_wrist_2_joint",      "left_arm_wrist_3_joint"],
    ))
    rclpy.spin_until_future_complete(node, fut)

    print("going to pre-contact!")
    fut = node.execute_traj(
        "both", np.array([3.5]),
        np.array([[
             1.56625366, -1.89681782, -2.23785257, -0.56585439,  0.59162313, -2.90597898,
            -1.76819355, -0.31465657,  2.04206449, -3.49937262,  0.81858134, -2.63544041,
        ]]))
    await_action_future(node, fut)

    print("executing trajectories ...")
    execute_trajectories(node, trajs)

    time.sleep(2.5)
    if with_slides:
        node.switch_slide("protocol_bag_2")

    print("waiting for force!")
    node.wait_for_force_change(1.45)

    print("FORCE CHANGE")
    if with_slides:
        time.sleep(2)
        node.switch_slide("protocol_bag_3")
        time.sleep(3.5)
    else:
        time.sleep(2)

    node.call_cli_sync(node.finger2srv["right_v2"], RollerGripperV2.Request(position=1.0))
    time.sleep(0.2)

    fut = node.execute_traj(
        "left", np.array([1.5, 3.0]),
        np.array([PRE_PLACE_LEFT, PLACE_LEFT]))
    await_action_future(node, fut)

    node.call_cli_sync(node.finger2srv["left_v2"], RollerGripperV2.Request(position=1.0))


# ═══════════════════════════════════════════════════════════════════════════
#  Public entry point called from tui_controller
# ═══════════════════════════════════════════════════════════════════════════

def run(state, node, bag_node: TrajectoryPublisher, mh2: MotionHelperV2):
    """
    Entry point called from a daemon thread in tui_controller.

    Installs a routing ask function so that every input() call shows
    the TUI modal and blocks until the operator answers Y/N + Enter.

    Prompt behaviour:
      "################### move? (y/N)"  →  modal, y=continue / other=retry
      "init pose? (Y/n)"                 →  modal, anything != "n" runs pose
    """
    global _ask_fn
    gate = state.bagopen_gate

    def routed_ask(prompt=""):
        prompt_str = prompt.strip() if isinstance(prompt, str) else str(prompt)
        state.add_log(f"  [bagopen] waiting for input: {prompt_str!r}")
        answer = gate.ask(prompt_str)
        state.add_log(f"  [bagopen] got: {answer!r}")
        return answer

    _ask_fn = routed_ask
    try:
        INITAL_POSE_TIME = 4
        arrays = load_trajectories(TRAJECTORIES_DIR)
        if not arrays:
            state.add_log("⚠  BAG OPEN: no trajectory files found")
            return

        bag_type = BagType.NORMAL
        state.add_log("→ BAG OPEN started (with_slides=True)")

        # Open both grippers at the start
        bag_node.call_cli_sync(bag_node.finger2srv["right_v2"],
                               RollerGripperV2.Request(position=0.6))
        bag_node.call_cli_sync(bag_node.finger2srv["left_v2"],
                               RollerGripperV2.Request(position=0.8))

        # ── "init pose?" confirmation ──────────────────────────────────────
        if _ask("init pose? (Y/n)").lower().strip() != "n":
            bag_node.initial_pose_new(dur=INITAL_POSE_TIME)
            time.sleep(0.4)

        if bag_node.with_slides:
            bag_node.switch_slide("protocol_bag_1")

        execute_opening(mh2, bag_node, arrays,
                        bag_type=bag_type, with_slides=bag_node.with_slides)

        if bag_node.with_slides:
            time.sleep(1)
            bag_node.switch_slide("protocol_11")
            bag_node.initial_pose_new(dur=INITAL_POSE_TIME)

        state.add_log("✓ BAG OPEN finished")

        # Publish True so tui_controller unlocks the gated slide
        msg      = Bool()
        msg.data = True
        node.pub_bagopen_done.publish(msg)
        state.add_log("  published True → /bag_open_done")

    except Exception as e:
        state.add_log(f"⚠  BAG OPEN exception: {e}")
    finally:
        _ask_fn = input                 # restore default
        with state._lock:
            state.bagopen_running = False
        if gate.waiting:
            gate.answer("n")            # unblock gate if sequence ended unexpectedly