"""Local collision-search steps, not pair-specific collision acceptance rules."""
import numpy as np
from tools.common_robust_proxy_penalty import RobustProxyPenaltySolver

class GeometryProgressPenaltySolver(RobustProxyPenaltySolver):
    def clearance_values(self,pairs):
        physical_pairs=[tuple(p[:2]) for p in pairs]
        distances,derivatives=super().clearance_values(physical_pairs)
        # Optional third item is an ephemeral local search goal derived from
        # current detailed penetration. Final acceptance NEVER reads this goal.
        required=np.array([p[2] if len(p)==3 else self.clearance for p in pairs])
        return distances-required+self.clearance,derivatives
