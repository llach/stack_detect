import time
import mujoco as mj
import threading
import mujoco_viewer
import stack_approach

from stack_approach.ik.joint import JointType as JT
from stack_approach.ik.common import add_frame_marker, mj2quat
from stack_approach.ik.robot_model import RobotModel

class RobotSim:

    qmax = 6.283
    umax = 360

    def __init__(self, tip_frame, with_vis=False) -> None:
        self.with_vis = with_vis
        self.thread_active = False
        
        self.model = mj.MjModel.from_xml_path(f"{stack_approach.__path__[0]}/assets/ur5_test.xml")
        self.data = mj.MjData(self.model)

        self.u2q = lambda x: x*(self.qmax/self.umax)
        self.q2u = lambda x: x*(self.umax/self.qmax)

         # create robot model
        links = [
            ("base_structure", JT.FIX),
            ("arm_base_link", JT.FIX),
            ("shoulder_link", JT.REV),
            ("upper_arm_link", JT.REV),
            ("forearm_link", JT.REV),
            ("wrist_1_link", JT.REV),
            ("wrist_2_link", JT.REV),
            ("wrist_3_link", JT.REV),
            ("hand_link", JT.FIX),
            ("hand_base", JT.FIX),
        ]

        # TODO publish this frame (maybe tune it also)
        self.ur5 = RobotModel(self.model, self.data, links=links, tip_frame=tip_frame)

        if self.with_vis:
            viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

            viewer.cam.azimuth      = -161
            viewer.cam.distance     = 1.56
            viewer.cam.elevation    = -10   
            viewer.cam.lookat       = [-0.04420583, -0.06552254,  0.84811352]

            self.viewer = viewer

            # self.viewer_thread = threading.Thread(target=self.run_viewer)
            # self.viewer_thread.start()
        
    def __del__(self):
        if hasattr(self, "viewer"): 
            if self.thread_active:
                self.thread_active = False
                self.viewer_thread.join()
            self.viewer.close()

    def update_robot_state(self, q):
        for name, qi in q.items():
               self.data.joint(name).qpos = qi
        mj.mj_step(self.model, self.data)

    def draw_goal(self, goal): 
        if self.with_vis and hasattr(self, "viewer"):
            add_frame_marker(goal, viewer=self.viewer, label="GOAL", scale=1, alpha=1)

    def render(self):
        if self.with_vis:
            print("hey")
            T, _, Ts = self.ur5.fk(fk_type="space")
            for name, T in Ts.items():
                add_frame_marker(T, viewer=self.viewer, label=name, scale=0.3, alpha=0.5)
        self.viewer.render()

    def run_viewer(self):
        self.thread_active = True
        while self.thread_active:
            print("hey2")
            T, _, Ts = self.ur5.fk(fk_type="space")
            for name, T in Ts.items():
                add_frame_marker(T, viewer=self.viewer, label=name, scale=0.3, alpha=0.5)

            self.viewer.render()
            time.sleep(0.1)
