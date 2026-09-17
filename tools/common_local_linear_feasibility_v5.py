"""Bounded sequential linear feasibility repair with independent acceptance.

Per-joint 0.01rad trust box limits only an optimizer update, not trajectory
acceptance. The linear program minimizes update infinity norm under unchanged
physical bounds and linearized fixed-target balls / detailed search planes.
"""
import numpy as np
from scipy.optimize import linprog


def solve(solver,targets,hands,initial,allowances,dt,arm,goals,max_iterations=20):
    q=initial.copy();n=len(q);ids=np.arange(arm*7,arm*7+6)
    active=np.ones((n,6),bool);active[:2]=active[-2:]=False;active=active.ravel()
    d1=np.kron(np.diff(np.eye(n),axis=0),np.eye(6));d2=np.kron(np.diff(np.eye(n),n=2,axis=0),np.eye(6))
    cfg=solver.config;step=min(cfg['maximum_joint_step_rad'],cfg['maximum_velocity_rad_s']*dt)
    acc=cfg['maximum_acceleration_rad_s2']*dt**2
    matrix=np.vstack((d1,d2));limits=np.r_[np.full(len(d1),step),np.full(len(d2),acc)]
    history=[];nv=int(active.sum())
    for it in range(max_iterations):
        base=q[:,ids].ravel();vel=matrix@base
        aa=[matrix[:,active],-matrix[:,active]];bb=[limits-vel,limits+vel]
        position_violation=[];geometry_violation=[]
        for f in range(2,n-2):
            pos,jac=solver.pose_jacobian(q[f],hands[f]);e=pos[arm]-targets[f,arm];length=np.linalg.norm(e)
            row=np.zeros(n*6);row[f*6:f*6+6]=(e/max(length,1e-30))@jac[arm*3:arm*3+3,ids]
            aa.append(row[active][None]);bb.append(np.array([allowances[f,arm]-length-1e-10]))
            position_violation.append(float(length-allowances[f,arm]))
            if f in goals:
                ds,js=solver.clearance_values(goals[f])
                for d,j in zip(ds,js):
                    row=np.zeros(n*6);row[f*6:f*6+6]=-j[ids]
                    aa.append(row[active][None]);bb.append(np.array([d-solver.clearance-1e-10]))
                    geometry_violation.append(float(solver.clearance-d))
        A=np.vstack(aa);b=np.concatenate(bb)
        # Minimize the size of a joint update; there is no representation input.
        A=np.c_[A,np.zeros(len(A))]
        A=np.vstack((A,np.c_[np.eye(nv),-np.ones(nv)],np.c_[-np.eye(nv),-np.ones(nv)]))
        b=np.r_[b,np.zeros(2*nv)];objective=np.r_[np.zeros(nv),1.]
        lower=np.maximum(np.tile(solver.lower[ids],n)[active]-base[active],-.01)
        upper=np.minimum(np.tile(solver.upper[ids],n)[active]-base[active],.01)
        fit=linprog(objective,A_ub=A,b_ub=b,bounds=[*zip(lower,upper),(0,.01)],method='highs',
                    options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
        row=dict(iteration=it,lp_success=bool(fit.success),message=str(fit.message),
                 maximum_position_excess_m=max(position_violation),maximum_geometry_plane_violation_m=max(geometry_violation,default=0))
        history.append(row);print('LINEAR_FEASIBILITY',row,flush=True)
        if not fit.success:break
        delta=np.zeros(n*6);delta[active]=fit.x[:-1];q[:,ids]+=delta.reshape(n,6)
        row['step_inf_rad']=float(fit.x[-1])
        if fit.x[-1]<1e-9:break
    return q,dict(history=history,iterations=len(history),acceptance='Independent complete raw/temporal/detailed-geometry checks required')
