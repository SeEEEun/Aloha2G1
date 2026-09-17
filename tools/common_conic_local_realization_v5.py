"""Method-blind sequential conic local realization with unchanged raw targets.

Uses isolated Clarabel 0.11.1, following its official Python problem format:
https://clarabel.org/stable/python/getting_started_py/
Only search approximations/slacks are introduced; full original acceptance is
independent. No robot asset, model option or installed environment is modified.
"""
from pathlib import Path
import sys
import numpy as np
from scipy import sparse
vendor=Path(__file__).resolve().parents[1]/'outputs/final_single_variable_ab/master_autonomous/local_redundancy_v5/solver_dependencies/clarabel_0_11_1'
sys.path.insert(0,str(vendor))
import clarabel
sys.path.remove(str(vendor))

def solve(solver,targets,hands,initial,allowances,dt,arm,goals,trust_radius=.01,max_iterations=100):
    q=initial.copy();n=len(q);ids=np.arange(arm*7,arm*7+6)
    active=np.ones((n,6),bool);active[:2]=active[-2:]=False;active=active.ravel();nv=int(active.sum())
    d1=np.kron(np.diff(np.eye(n),axis=0),np.eye(6));d2=np.kron(np.diff(np.eye(n),n=2,axis=0),np.eye(6))
    cfg=solver.config;step=min(cfg['maximum_joint_step_rad'],cfg['maximum_velocity_rad_s']*dt)
    matrix=np.vstack((d1,d2));limits=np.r_[np.full(len(d1),step),np.full(len(d2),cfg['maximum_acceleration_rad_s2']*dt**2)]
    history=[];radius=trust_radius
    def evaluate(v,build=False):
        A=[];b=[];cones=[];violations=[]
        for f in range(2,n-2):
            pos,jac=solver.pose_jacobian(v[f],hands[f]);e=pos[arm]-targets[f,arm]
            allowance=allowances[f,arm]-1e-8
            violations.append(1e3*(np.linalg.norm(e)-allowance))
            if build:
                block=np.zeros((4,nv+1));block[0,-1]=-1.
                full=np.zeros((3,n*6));full[:,f*6:f*6+6]=-1e3*jac[arm*3:arm*3+3,ids]
                block[1:,:nv]=full[:,active]
                A.append(block);b.append(np.r_[allowance*1e3,e*1e3]);cones.append(clarabel.SecondOrderConeT(4))
            if f in goals:
                ds,js=solver.clearance_values(goals[f])
                for dist,j in zip(ds,js):
                    violations.append(1e3*(solver.clearance-dist)+1e-6)
                    if build:
                        row=np.zeros(n*6);row[f*6:f*6+6]=-1e3*j[ids]
                        A.append(np.r_[row[active],0][None]);b.append(np.array([1e3*(dist-solver.clearance)-1e-6]));cones.append(clarabel.NonnegativeConeT(1))
        return A,b,cones,float(max(0.,max(violations)))
    for it in range(max_iterations):
        base=q[:,ids].ravel();A,b,cones,metric=evaluate(q,True)
        if metric<=1e-8:break
        vm=matrix@base
        lower=np.maximum(np.tile(solver.lower[ids],n)[active]-base[active],-radius)
        upper=np.minimum(np.tile(solver.upper[ids],n)[active]-base[active],radius)
        linear=np.vstack((np.c_[matrix[:,active],np.zeros(len(matrix))],np.c_[-matrix[:,active],np.zeros(len(matrix))],
            np.c_[np.eye(nv),np.zeros(nv)],np.c_[-np.eye(nv),np.zeros(nv)],np.r_[np.zeros(nv),-1][None]))
        rhs=np.r_[limits-vm,limits+vm,upper,-lower,0.]
        A.append(linear);b.append(rhs);cones.append(clarabel.NonnegativeConeT(len(rhs)))
        settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=150
        settings.tol_gap_abs=settings.tol_gap_rel=settings.tol_feas=1e-10
        fit=clarabel.DefaultSolver(sparse.diags(np.r_[np.full(nv,2e-6),0.],format='csc'),np.r_[np.zeros(nv),1.],
            sparse.csc_matrix(np.vstack(A)),np.concatenate(b),cones,settings).solve()
        status=str(fit.status)
        row=dict(iteration=it,solver_status=status,maximum_violation_mm=metric,trust_radius_rad=radius)
        history.append(row)
        if status not in ('Solved','AlmostSolved','InsufficientProgress'):break
        delta=np.zeros(n*6);delta[active]=np.array(fit.x[:nv]);delta=delta.reshape(n,6)
        row['linearized_position_slack_mm']=float(fit.x[-1]);row['delta_inf_rad']=float(np.abs(delta).max())
        best=None
        for fraction in (1.,.5,.25,.125,.0625,.03125,.015625,.0078125):
            trial=q.copy();trial[:,ids]+=fraction*delta;score=evaluate(trial)[3]
            if score<metric-1e-11 or score<=1e-8:best=(trial,score,fraction);break
        row['accepted']=best is not None
        if it%5==0:print('COMMON_CONIC_REALIZATION',row,flush=True)
        if best is None:
            radius*=.5
            if radius<1e-7:break
            continue
        q,score,fraction=best;row.update(new_maximum_violation_mm=score,line_fraction=fraction)
        if score<=1e-8:break
    return q,dict(iterations=len(history),history=history,raw_targets_modified=False,
        dependency='clarabel==0.11.1 isolated --no-deps install',acceptance_requires_independent_requalification=True)
