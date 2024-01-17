import qpsolvers
import numpy as np
import tf_transformations as tf

from .common import mj2quat, adjoint, invert_T
from .joint import JointType, Joint

class RobotModel:
    clip_threshold = 0.1

    def __init__(self, model, data, links, tip_frame=None):
        # we store mj objects to access data (e.g. joint pos)
        self.model = model
        self.data = data
        self.links = links
        self.tip_frame = tip_frame

        # build joint chain from base frame to last arm frame
        self.joints = []
        for link, jnt_t in self.links:
            body = self.model.body(link)
            if jnt_t == JointType.FIX:
                jn, axis, limit = link, [0,0,0], [0,0]
            else:
                joint = self.model.joint(body.jntadr[0]) # assume 1-to-1 relationships of bodies and joints
                jn, axis, limit = joint.name, joint.axis, joint.range

            self.joints.append(Joint(
                name=jn,
                jnt_type=jnt_t,
                pos=body.pos,
                quat=mj2quat(body.quat),
                axis=axis,
                limit=limit,
                parent=None if self.N==0 else self.joints[-1]
            ))

        # append (optional) tip frame
        if self.tip_frame:
            self.joints.append(Joint(
                name=tip_frame[0],
                pos=tip_frame[1],
                quat=tip_frame[2],
                limit=[0,0],
                axis=[0,0,0],
                jnt_type=JointType.FIX,
                parent=self.joints[-1]
            ))

        # scale joint contribution to J, e.g. disable joint i by setting jnt_scale[i]=0
        self.jnt_scale = np.repeat(1, self.N)

    @property
    def N(self): return len(self.joints)

    @property
    def q(self): return np.array([self.data.joint(jnt.name).qpos[0] if jnt.jnt_type is not JointType.FIX else 0.0 for jnt in self.joints])

    @property
    def joint_names(self): return [jnt.name for jnt in self.joints]

    @property
    def limits(self): return np.array([joint.limit for joint in self.joints])

    @property
    def mins(self): return self.limits[:,0]

    @property
    def maxs(self): return self.limits[:,1]

    def make_qdict(self, q): return {jnt.name: qi for jnt, qi in zip(self.joints, q)} 

    def clip_q(self, q): return np.clip(q, self.limits[:,0], self.limits[:,1])
    def random_valid(self): return np.random.uniform(low=self.mins, high=self.maxs)

    def get_mj_qs(self, qs):
        q, qdict = np.zeros(self.model.njnt-1), self.make_qdict(qs) # TODO the -1 ignores the freejoint. annoyting!
        for i in range(self.model.njnt):
            jnt = self.data.joint(i)
            if jnt.name not in qdict: continue
            q[i] = qdict[jnt.name]
        return q
    
    def disable_joint(self, jnt_name):
        jidx = self.joint_names.index(jnt_name)
        self.jnt_scale[jidx] = 0

    def fk(self, q=None, fk_type="space", until_joint=None):
        """
        q: set of joint values to use for FK. if None, latest values are used.
        fk_type: forward kinematics calculation type to use. not all yield J
        until_joint: name of latest joint to inculde in J; default are all joints
        """
        assert fk_type in ["mult", "space", "body"], f"unknown FK type {fk_type}"
        if until_joint is not None: assert until_joint in self.joint_names, f"{until_joint} in self.joint_names"
        if q is None: q = self.q

        # Jacobian has dimensionality 6×N 
        J = np.zeros([6, self.N])

        # we store all frame origins for viz purposes, Ts[-1] is T[base → EEF]
        Ts = np.stack([np.eye(4) for _ in range(self.N)])

        if fk_type == "mult":
            """ we simply multiply all transforms; this won't yield a Jacobian and is meant for verification purposes
            """
            for joint, q in zip(self.joints, q): 
                T = np.dot(T, joint(q))
                Ts.append(T)
            J = None # set Jacobian to None; we rather want errors than annoying math debugging here

        elif fk_type == "space":
            """ we calculate twists for each joint from the base to the tip frame, expressed in base frame coordinates. c.f. Lynch&Park p.178
            """

            for idx, (joint, q) in enumerate(zip(self.joints, q)):
                # Ts[i] is static joint offset * movement about joint axis → transform from base to current frame
                Ts[idx,:] = Ts[max(0,idx-1),:].dot(joint(q))

                if joint.jnt_type != JointType.FIX: # fixed joints do not contribute to J; their static tf is incorporated in the subsequent T

                    # the twist is simply determined by the joint type and its moving axis. thinking about entries of J as describing the change of cartesian EEF pose based on the joint angle, it's obvious that one joint can only affect one dimension of the pose (w.r.t. its parent frame and assuming no other joints afterwards move).
                    base_twist = np.block([np.zeros(3), joint.axis]) if joint.jnt_type == JointType.REV else np.block([joint.axis, np.zeros(3)]) 

                    # the adjoint of Ts[i] transforms base frame twists to the current joint frame
                    Ad_T = adjoint(Ts[idx,:])

                    # express twist in current joint frame
                    twist = Ad_T.dot(base_twist)

                    J[:,idx] += self.jnt_scale[idx] * twist # insert twist, optionally scale it
                
                # check if we've reached the last joint
                if until_joint is not None and joint.name == until_joint: break

        elif fk_type == "body":
            """ like the space J, but starting from the tip frame, going back to the base frame. c.f. Lynch&Park p.183
            """

            for idx, (joint, q) in enumerate(zip(self.joints[::-1], q[::-1])): # note the reversed lists
                idx = self.N-idx-1 # reverse index too

                Ad_T = adjoint(Ts[min(self.N-1, idx+1),:], inverse=True)

                # here Ts[idx-1,:] represents the tranform from the previous joint frame to the tip frame
                # hence we pre-multiply T_i(theta_i) to get the 
                Ts[idx,:] = joint(q).dot(Ts[min(self.N-1, idx+1),:])

                if joint.jnt_type != JointType.FIX:
                    base_twist = np.block([np.zeros(3), joint.axis]) if joint.jnt_type == JointType.REV else np.block([joint.axis, np.zeros(3)]) 

                    # twist of joint motion in EEF frame
                    twist = Ad_T.dot(base_twist)
                    J[:,idx] += self.jnt_scale[idx] * twist

            # Ts[i] should correspond to the T[base → joint_i+1]
            # currently, T[0] is T[base → EEF], which should be T[-1], so we roll the array
            Ts = np.roll(Ts, -1, axis=0)

            # all transforms (except the last) need to be transformed to base coords
            for i in np.arange(self.N-1):
                # current T[i]s transform joint_i to EEF, so T[base → EEF]@T[i]^-1 yields T[base → joint_i]
                Ts[i,:] = Ts[-1,:]@invert_T(Ts[i,:])

            # all twists in the Jacobian are expressed in EEF frame
            # we use the inverse Adjoint matrix from T[base → EEF] to transform them into base frame twists
            R = Ts[-1,:3,:3] # R[base → EEF]
            Ad_EEF = np.block([[R, np.zeros((3, 3))], [np.zeros((3, 3)), R]])
            J = Ad_EEF.dot(J)

        return Ts[-1,:], J, dict(zip(self.joint_names, Ts))

    def ik(self, tasks):
        if isinstance(tasks, tuple): tasks = [tasks]

        """ initialization
        """
        N = np.eye(self.N) # nullspace projector of prior task
        Js = np.zeros((0,self.N)) # Jacobians of previous tasks
        qdot = np.zeros(self.N) # initial joint velocities

        for p, (J, err) in enumerate(tasks):
            """ J is (part of) the Jacobian, and err the error (in cartesian space) to minimize in ||J*qdot - err||
            """
            err = np.array(err)
            if len(J.shape)==1: J = np.expand_dims(J, axis=0)

            def invert_smooth_clip(s):
                return s/(self.clip_threshold**2) if s < self.clip_threshold else 1./s

            # calculate SVD of (nullspace-projected) Jacobian for the inversion
            # dim(J) = m×N (N = number of joints)
            # dim(U) = m×m (task space dimensionality)
            # dim(Vt) = N×N
            # dim(S) = min(m,N) → diagonal matrix in vector form
            U, S, Vt = np.linalg.svd(J.dot(N))

            # first invert sigma
            rank = min(U.shape[0], Vt.shape[0])
            for i in range(rank): S[i] = invert_smooth_clip(S[i])

            # calculate inverse of Jacobian after Nullspace projection
            # TODO why V only up to rank?
            # JNinv = Vt.T[:,:rank].dot(np.diag(S).dot(U.T))

            # calculate change in qdot
            # TODO elaborate more
            # _qdot = JNinv.dot(err - J.dot(qdot))
            # qdot += _qdot

            # TODO how is this equivalent to the above?
            qdot += np.dot(Vt.T[:, 0:rank], S * U.T.dot(np.array(err) - J.dot(qdot))).reshape(qdot.shape)

            # compute new nullspace projector
            Js = np.vstack([Js, J])
            U, S, Vt = np.linalg.svd(Js)
            accepted_singular_values = (S > 1e-3).sum() # corresponding to index k in script
            # print(f"task {p}: rank(N)={self.N-accepted_singular_values}")
            VN = Vt[accepted_singular_values:].T
            N = VN.dot(VN.T)

        return qdot, N

    def solve_qp(self, tasks):
        """Solve tasks (J, ub, lb) of the form lb ≤ J dq ≤ ub
           using quadratic optimization: https://pypi.org/project/qpsolvers"""
           
        if type(tasks)==tuple: tasks = [tasks]

        maxM = np.amax([task[0].shape[0] for task in tasks]) # max task dimension
        sumM = np.sum([task[0].shape[0] for task in tasks]) # sum of all task dimensions
        usedM = 0
        # allocate arrays once
        G, h = np.zeros((2*sumM, self.N + maxM)), np.zeros(2*sumM)
        P = np.identity(self.N+maxM)
        P[self.N:, self.N:] *= 1.0  # use different scaling for slack variables?
        q = np.zeros(self.N + maxM)

        # joint velocity bounds + slack bounds
        upper = np.hstack([np.minimum(0.1, self.maxs - self.q), np.zeros(maxM)])
        lower = np.hstack([np.maximum(-0.1, self.mins - self.q), np.full(maxM, -np.infty)])

        # fallback solution
        dq = np.zeros(self.N)

        def add_constraint(A, bound):
            G[usedM:usedM+M, :N] = A
            G[usedM:usedM+M, N:N+M] = np.identity(M)  # allow (negative) slack variables
            h[usedM:usedM+M] = bound
            return usedM + M

        for idx, task in enumerate(tasks):
            try:  # inequality tasks are pairs of (J, ub, lb=None)
                J, ub, lb = task
            except ValueError:  # equality tasks are pairs of (J, err)
                J, ub = task
                lb = ub  # turn into inequality task: err ≤ J dq ≤ err
            J = np.atleast_2d(J)
            M, N = J.shape

            # augment G, h with current task's constraints
            oldM = usedM
            usedM = add_constraint(J, ub)
            if lb is not None:
                usedM = add_constraint(-J, -lb)

            result = qpsolvers.solve_qp(P=P[:N+M, :N+M], q=q[:N+M],
                                        G=G[:usedM, :N+M], h=h[:usedM], A=None, b=None,
                                        lb=lower[:N+M], ub=upper[:N+M],
                                        solver="osqp")
            
            if result is None:
                # print("{}: failed  ".format(idx), end='')
                usedM = oldM  # ignore subtask and continue with subsequent tasks
            else: # adapt added constraints for next iteration
                dq, slacks = result[:N], result[N:]
                # print("{}:".format(idx), slacks, " ", end='')
                G[oldM:usedM,N:N+M] = 0
                h[oldM:oldM+M] += slacks
                if oldM+M < usedM:
                    h[oldM+M:usedM] -= slacks
        # print()
        self.nullspace = np.zeros((self.N, 0))
        return dq, self.nullspace
    
    @staticmethod
    def stack(tasks):
        """Combine all tasks by stacking them into a single Jacobian"""
        Js, errs = zip(*tasks)
        return np.vstack(Js), np.hstack(errs)
    
    def position_task(self, T_tgt, T_cur, J, scale=1.0):
        """Move eef towards a specific target point in base frame"""
        return J[:3], scale*(T_tgt[0:3, 3]-T_cur[0:3, 3])

    def orientation_task(self, T_tgt, T_cur, J, scale=1.0):
        """Move eef into a specific target orientation in base frame"""
        delta = np.identity(4)
        delta[0:3, 0:3] = T_cur[0:3, 0:3].T.dot(T_tgt[0:3, 0:3])
        angle, axis, _ = tf.rotation_from_matrix(delta)
        # transform rotational velocity from end-effector into base frame orientation (only R!)
        return J[3:], scale*(T_cur[0:3, 0:3].dot(angle * axis))

    def pose_task(self, T_tgt, T_cur, J, scale=(1.0, 1.0)):
        """Perform position and orientation task with same priority"""
        return self.stack([self.position_task(T_tgt, T_cur, J, scale=scale[0]),
                           self.orientation_task(T_tgt, T_cur, J, scale=scale[1])])