import numpy as np
import mujoco as mj
import tf_transformations as tf

from .joint import JointType

""" Math 
"""

def cost_derivative(cost_fn, robot, q=None, delta=0.001, **kw):
    if q is None: q = robot.q

    # initialize derivative with zeros, fixed joints will have dh/dq = 0
    dhdq = np.zeros(robot.N)

    # cost for base
    cost = cost_fn(robot, q=q, **kw)

    for i in range(robot.N):
        # skip fixed joints
        if robot.joints[i].jnt_type == JointType.FIX: continue

        # add small delta to q_i
        dhdqi = q.copy()
        dhdqi[i] += delta

        # respect joint limits
        dhdqi = robot.clip_q(dhdqi)

        # calculate derivative of cost wrt q_i
        dcost = cost_fn(robot=robot, q=dhdqi, **kw)
        dhdq[i] = (dcost-cost)/delta

    return dhdq

def jl_cost(robot, q):
    cs = [jl_cost_fn(qi, joint.limit) for qi, joint in zip(q, robot.joints)]
    return np.sum(cs)
    
def jl_cost_fn(q, lims, k=10**-12, pen_ratio=0.05):
    q = np.clip(q, *lims)

    pen_size = (lims[1]-lims[0])*pen_ratio
    pen_min = lims[0] + pen_size
    pen_max = lims[1] - pen_size

    if q < pen_min: 
        return min(k/(q-lims[0]+10**-24),  2)
    if q > pen_max:
        return min(-k/(q-lims[1]-10**-24), 2)
    return 0

def trafo(t=[0,0,0], q=[0,0,0,1]):
    T = tf.quaternion_matrix(q)
    T[:3,3] = t
    return T

def transform_point(T, p):
    if type(p) == list: p = np.array(p)

    if T.shape == (3,3): # Rotation matrix to homogeneous transform
        T = np.block([[T,np.zeros((3,1))], [0,0,0,1]])
    if p.shape == (3,):
        p = np.concatenate([p,[1]])
    return T.dot(p)[:3]

def hat(p):
    return np.array([[0, -p[2], p[1]],
                        [p[2], 0, -p[0]],
                        [-p[1], p[0], 0]])

def adjoint(T, inverse=False):
    if T.shape == (4, 4):
        R = T[0:3, 0:3]
        p = T[0:3, 3]
    elif T.shape == (3, 3):
        R = T
        p = np.zeros(3)
    else:
        R = np.identity(3)
        p = T
    if not inverse:
        return np.block([[R, hat(p).dot(R)], [np.zeros((3, 3)), R]])
    else:
        return np.block([[R.T, R.T.dot(hat(-p))], [np.zeros((3, 3)), R.T]])

def invert_T(T):
    R = T[:3,:3]
    t = T[:3, 3]

    Rinv = R.T
    Tinv = np.eye(4)

    Tinv[:3,:3] = Rinv
    Tinv[:3, 3] = -Rinv@t

    return Tinv

""" MuJoCo helpers
"""

def mj2quat(mjq): return np.concatenate([mjq[1:], [mjq[0]]])
def quat2mj(quat): return np.concatenate([[quat[3]], quat[:3]])

def set_subset_q(model, q, qdes = {}):
    for i in range(len(q)):
        jnt_name = model.joint(i).name
        if jnt_name not in qdes: continue
        q[i] = qdes[jnt_name]
    return q

def clip_q(model, q):
    for i in range(len(q)): q[i] = np.clip(q[i], *model.joint(i).range)
    return q

def add_box_marker(pos, viewer, dims=[0.02, 0.02, 0.02], color=[0, 1, 0.8, 0.7], **kw):
    if pos.shape == (4,4): pos = pos[:3,3]

    viewer.add_marker(
            pos=pos,
            type=mj.mjtGeom.mjGEOM_BOX, 
            size=dims,
            rgba=color,
            **kw
        )

def add_frame_marker(T, viewer, cyl_len=0.07, cyl_rad=0.01, label=None, scale=1.0, alpha=1.0, **kw):
    size = scale*np.array([cyl_rad, cyl_rad, cyl_len])

    Ts = np.repeat([T], 3, axis=0)
    Raligns = [
        tf.quaternion_matrix([0,0.707,0,0.707]),
        tf.quaternion_matrix([-0.70710678, 0, 0, 0.70710678]),
        tf.quaternion_matrix([0,0,0,1])
    ]
    TR = np.matmul(Ts, Raligns)

    toff = np.array(3*[np.eye(4)])
    toff[0,2,3] = scale*cyl_len
    toff[1,2,3] = scale*cyl_len
    toff[2,2,3] = scale*cyl_len
    TR = np.matmul(TR, toff)

    colors = [
        [1,0,0,alpha],
        [0,1,0,alpha],
        [0,0,1,alpha]
    ]

    for i in range(TR.shape[0]):
        kww = kw.copy()
        if i == 0 and label is not None: kww |= {"label": label}

        viewer.add_marker(
            pos=TR[i,:3,3],
            mat=TR[i,:3,:3],
            type=mj.mjtGeom.mjGEOM_CYLINDER, 
            size=size,
            rgba=colors[i],
            **kww
        )
