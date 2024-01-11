import numpy as np


class Robot:
    gripper_state = "closed"
    name = ""

    def __init__(self, name):
        self.name = name

    def close_gripper(self):
        self.gripper_state = "closed"

    def open_gripper(self):
        self.gripper_state = "open"

    def get_joints(self):
        return np.random.randint(10, size=7).tolist()
