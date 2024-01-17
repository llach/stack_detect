import numpy as np
import tf_transformations as tf

from enum import Enum
from dataclasses import dataclass

class JointType(Enum):
    FIX=0
    PRIS=1
    REV=2

@dataclass
class Joint:
    name: str
    jnt_type: JointType
    pos: np.array
    quat: np.array
    limit: np.array
    axis: np.array
    parent: str = None

    def __post_init__(self):
        self.T = np.eye(4)
        self.set_R(tf.quaternion_matrix(self.quat)[:3,:3])
        self.set_t(self.pos)

    def __call__(self, q=None):
        if self.jnt_type != JointType.FIX: assert q is not None, "q is not None"

        if self.jnt_type == JointType.REV:
            R = tf.rotation_matrix(q, self.axis)
            return self.T@R
        elif self.jnt_type == JointType.PRIS:
            H = np.eye(4)
            H[:3,3] = q*self.axis
            return self.T@H
        elif self.jnt_type == JointType.FIX: return self.T

    @property
    def t(self):
        return self.T[:3,3]

    @property
    def R(self):
        return self.T[:3,:3]

    def set_t(self, t): self.T[:3,3]=t
    def set_R(self, R): self.T[:3,:3]=R