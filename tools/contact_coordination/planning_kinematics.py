"""Geometry-only adapter over the existing G1 model, never a runtime robot.

The legacy assign() executes force/constraint dynamics for every IK residual.
Here the same qpos, body transforms and Jacobian prerequisites are computed
without those unused dynamics. Collision admission remains in runtime_hulls.
"""
import numpy as np
import mujoco
from tools.doll_handoff_retargeting.models import G1Kinematics as ExistingG1


class G1Kinematics(ExistingG1):
    def __init__(self, common, scene):
        self.query_statistics = dict(assign_calls=0, kinematics_calls=0)
        super().__init__(common, scene)
        self.named_qpos = {
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j):
            int(self.model.jnt_qposadr[j]) for j in range(self.model.njnt)}

    def forward_kinematics(self):
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        self.query_statistics['kinematics_calls'] += 1

    def assign(self, arm_q, left_hand=None, right_hand=None):
        self.query_statistics['assign_calls'] += 1
        self.data.qpos[:] = self.stand_qpos
        self.data.qpos[self.arm_qpos_ids] = np.asarray(arm_q, dtype=np.float64)
        for side, values in [('left', left_hand), ('right', right_hand)]:
            if values is not None:
                self.data.qpos[self.hand_qpos_ids[side]] = np.asarray(values, dtype=np.float64)
        self.data.qvel[:] = 0.
        self.forward_kinematics()

    def wrist_pose(self, side):
        # Rotations come directly from the same MuJoCo kinematics call.
        # Endpoint SE(3), finite-state and collision validation remain external.
        body = self.wrist_ids[side]
        result = np.eye(4)
        result[:3, :3] = self.data.xmat[body].reshape(3, 3)
        result[:3, 3] = self.data.xpos[body]
        return result
