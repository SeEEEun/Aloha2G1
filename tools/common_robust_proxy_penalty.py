"""Numerically consistent proxy distances for optimizer penalties only.

The unchanged detailed collision classifier remains the acceptance authority.
No model option persists beyond this distance query, including exceptions.
"""
import mujoco
from tools.final_common_position_solver import CommonPositionSolver

class RobustProxyPenaltySolver(CommonPositionSolver):
    def clearance_values(self,pairs):
        before=int(self.g1.model.opt.disableflags)
        try:
            self.g1.model.opt.disableflags=before|int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
            return super().clearance_values(pairs)
        finally:
            self.g1.model.opt.disableflags=before
