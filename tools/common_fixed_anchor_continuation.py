"""Method-blind bounded trajectory feasibility restoration.

Acceptance-ball and temporal violations are explicit residuals, not implicit
smoothness preferences. Final numerical/geometry gates are independently tested.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix,csr_matrix,vstack,eye,kron,diags


def restore(solver,targets,hands,initial,dt,allowances,collision_pairs=None,max_nfev=160,fixed_mask=None,fixed_values=None):
    n=len(initial);size=n*14;cfg=solver.config
    d1=kron(diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n)),eye(14),format='csr')
    d2=kron(diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n)),eye(14),format='csr')
    indices=np.concatenate([np.tile(np.arange(f*14,(f+1)*14),6) for f in range(n)])
    rows=np.repeat(np.arange(n*6),14)
    step_bound=min(cfg['maximum_joint_step_rad'],cfg['maximum_velocity_rad_s']*dt)-1e-5
    norm_bound=cfg['maximum_step_norm_rad']-1e-5
    acc_bound=cfg['maximum_acceleration_rad_s2']*dt*dt-1e-5
    pairs=collision_pairs or {};cached=None;result=None
    def compute(x):
        nonlocal cached,result
        if cached is not None and np.array_equal(cached,x):return result
        q=x.reshape(n,14);pj=[];pr=[];cr=[];cj=[];cridx=[];ccidx=[]
        for f in range(n):
            pos,jac=solver.pose_jacobian(q[f],hands[f]);error=pos-targets[f]
            # Distance to each unchanged raw-target acceptance ball.
            for side in range(2):
                e=error[side];length=np.linalg.norm(e);radius=allowances[f,side]-1e-6
                if length>radius:
                    u=e/length;derivative=(1-radius/length)*np.eye(3)+(radius/length)*np.outer(u,u)
                    pr.extend((1-radius/length)*e);pj.extend(derivative@jac[side*3:side*3+3])
                else:pr.extend(np.zeros(3));pj.extend(np.zeros((3,14)))
            if f in pairs:
                distances,derivatives=solver.clearance_values(pairs[f])
                for distance,derivative in zip(distances,derivatives):
                    ri=len(cr);active=distance<solver.clearance
                    cr.append(min(distance-solver.clearance,0.))
                    cj.extend(derivative if active else np.zeros(14));cridx.extend([ri]*14);ccidx.extend(range(f*14,(f+1)*14))
        pm=coo_matrix((np.array(pj).ravel(),(rows,indices)),shape=(n*6,size)).tocsr()
        delta=(d1@x).reshape(n-1,14);acc=d2@x
        scalar=np.maximum(np.abs(delta.ravel())-step_bound,0.)
        sa=(np.abs(delta.ravel())>step_bound)*np.sign(delta.ravel())
        accel=np.maximum(np.abs(acc)-acc_bound,0.);aa=(np.abs(acc)>acc_bound)*np.sign(acc)
        norms=np.linalg.norm(delta,axis=1);nr=np.maximum(norms-norm_bound,0.)
        unit=delta/np.maximum(norms[:,None],1e-30)*(norms>norm_bound)[:,None]
        rr=np.repeat(np.arange(n-1),28)
        cc=np.concatenate([np.arange(f*14,(f+2)*14) for f in range(n-1)])
        vv=np.hstack((-unit,unit)).ravel();nm=coo_matrix((vv,(rr,cc)),shape=(n-1,size)).tocsr()
        # Feasibility residuals dominate a tiny regularizer. This never declares
        # success: exact raw, temporal and detailed collision checks follow.
        residual=np.r_[100*np.array(pr),scalar,nr,accel,100*np.array(cr),1e-5*(d1@x),1e-7*(x-initial.ravel())]
        cm=coo_matrix((cj,(cridx,ccidx)),shape=(len(cr),size)).tocsr()
        matrix=vstack([100*pm,diags(sa)@d1,nm,diags(aa)@d2,100*cm,1e-5*d1,1e-7*eye(size)],format='csr')
        cached=x.copy();result=residual,matrix;return result
    mask=np.zeros_like(initial,dtype=bool) if fixed_mask is None else np.asarray(fixed_mask,dtype=bool)
    fixed=initial.copy() if fixed_values is None else np.asarray(fixed_values).copy()
    active=~mask.ravel()
    def expand(y):
        value=fixed.ravel().copy();value[active]=y;return value
    fit=least_squares(lambda y:compute(expand(y))[0],initial.ravel()[active],jac=lambda y:compute(expand(y))[1][:,active],
        bounds=(np.tile(solver.lower,n)[active],np.tile(solver.upper,n)[active]),max_nfev=max_nfev,
        ftol=1e-10,xtol=1e-10,gtol=1e-10,tr_solver='lsmr',tr_options={'maxiter':200})
    return expand(fit.x).reshape(n,14),dict(nfev=int(fit.nfev),status=int(fit.status),cost=float(fit.cost),optimality=float(fit.optimality))

