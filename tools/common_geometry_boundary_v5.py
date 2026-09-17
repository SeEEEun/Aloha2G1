"""Local detailed-geometry boundary search; a solver aid, not acceptance.

Discover controlling joints from the collision-distance Jacobian, bracket the
actual common classifier's clear side, then build a local tangent constraint.
No link whitelist or method/episode argument exists.
"""
import numpy as np
from tools.common_robust_proxy_penalty import RobustProxyPenaltySolver


class GeometryBoundarySolver(RobustProxyPenaltySolver):
    def __init__(self, *args):
        super().__init__(*args)
        self.boundaries = {}

    def pose_jacobian(self, q, hands):
        self.current_q = np.array(q).copy()
        return super().pose_jacobian(q, hands)

    def clearance_values(self, pairs):
        values, rows = [], []
        for p in pairs:
            key=tuple(p[:2])
            if key not in self.boundaries:
                d,j=super().clearance_values([key]); values.append(d[0]); rows.append(j[0]); continue
            b=self.boundaries[key]
            grad=np.array(b['gradient']); point=np.array(b['point'])
            # Search-plane units are scaled radians. All final classifications
            # still use detailed meshes and the frozen 10 micrometre rule.
            values.append(self.clearance + .04 * grad @ (self.current_q-point))
            rows.append(.04 * grad)
        return np.array(values), np.array(rows).reshape(-1,14)

    def discover(self,q,hands,pair):
        self.pose_jacobian(q,hands)
        _,jj=super().clearance_values([pair]); j=jj[0]
        ids=np.flatnonzero(np.abs(j)>1e-9)
        control=int(ids[np.argmax(np.abs(j[ids]))]); direction=float(np.sign(j[control]))
        evidence=[]
        def clear(x):
            rr=self.collision.inspect(x,*hands)
            bad=[r for r in rr if tuple(r['geom_pair'])==tuple(pair) and r['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY')]
            return not bad
        def boundary(base):
            lo=0.;hi=None
            for delta in (.005,.01,.02,.04,.08,.16,.32):
                x=base.copy();x[control]+=direction*delta
                if not self.lower[control]<x[control]<self.upper[control]: break
                if clear(x):hi=delta;break
            if hi is None:raise RuntimeError('No clear bracket in bounded common joint displacement search')
            for _ in range(18):
                mid=(lo+hi)/2;x=base.copy();x[control]+=direction*mid
                if clear(x):hi=mid
                else:lo=mid
            x=base.copy();x[control]+=direction*hi
            evidence.append(dict(base=base.tolist(),clear_q=x.tolist(),bracket_width_rad=hi-lo))
            return x
        center=boundary(q);gradient=np.zeros(14);gradient[control]=direction
        for k in ids:
            if k==control:continue
            left=q.copy();right=q.copy();left[k]-=.005;right[k]+=.005
            a=boundary(left);b=boundary(right)
            gradient[k]=-direction*(b[control]-a[control])/.01
        record=dict(point=center.tolist(),gradient=gradient.tolist(),control_joint=control,
                    influenced_joints=ids.tolist(),evidence=evidence,
                    interpretation='Local geometry search tangent only; reclassification mandatory')
        self.boundaries[tuple(pair)]=record
        return record
