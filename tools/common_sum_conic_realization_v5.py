"""Bounded method-blind elastic SOCP search with hard physical linear bounds.

Per-frame elastic search variables avoid a maximum-residual plateau; they are
never acceptance allowances. Final nonlinear, temporal and geometry gates are
independent and unchanged.
"""
import numpy as np
from scipy import sparse
from tools.common_conic_trust_realization_v6 import clarabel

def solve(solver,targets,hands,initial,allowances,dt,arm,trust_radius=.15,max_iterations=500):
    q=initial.copy();n=len(q);ids=np.arange(arm*7,arm*7+6)
    active=np.ones((n,6),bool);active[:2]=active[-2:]=False;active=active.ravel();nv=int(active.sum());nz=n-4
    d1=np.kron(np.diff(np.eye(n),axis=0),np.eye(6));d2=np.kron(np.diff(np.eye(n),n=2,axis=0),np.eye(6))
    cfg=solver.config;step=min(cfg['maximum_joint_step_rad'],cfg['maximum_velocity_rad_s']*dt)
    matrix=np.vstack((d1,d2));limits=np.r_[np.full(len(d1),step),np.full(len(d2),cfg['maximum_acceleration_rad_s2']*dt**2)]
    radius=trust_radius;history=[]
    def evaluate(v,build=False):
        A=[];b=[];cones=[];pv=[]
        for f in range(2,n-2):
            p,J=solver.pose_jacobian(v[f],hands[f]);e=p[arm]-targets[f,arm];allow=allowances[f,arm]-1e-8
            pv.append(max(0.,1000*(np.linalg.norm(e)-allow)))
            if build:
                block=np.zeros((4,nv+nz));block[0,nv+f-2]=-1
                full=np.zeros((3,n*6));full[:,f*6:f*6+6]=-1000*J[arm*3:arm*3+3,ids]
                block[1:,:nv]=full[:,active];A.append(block);b.append(np.r_[1000*allow,1000*e]);cones.append(clarabel.SecondOrderConeT(4))
        physical=100*np.maximum(np.abs(matrix@v[:,ids].ravel())-limits,0)
        return A,b,cones,float(np.linalg.norm(np.r_[pv,physical])),max(pv,default=0.),float(np.max(physical,initial=0))
    for it in range(max_iterations):
        A,b,cones,merit,maxpos,maxphys=evaluate(q,True)
        if max(maxpos,maxphys)<=1e-8:break
        base=q[:,ids].ravel();vm=matrix@base
        lower=np.maximum(np.tile(solver.lower[ids],n)[active]-base[active],-radius)
        upper=np.minimum(np.tile(solver.upper[ids],n)[active]-base[active],radius)
        linear=np.vstack((np.c_[matrix[:,active],np.zeros((len(matrix),nz))],np.c_[-matrix[:,active],np.zeros((len(matrix),nz))],
            np.c_[np.eye(nv),np.zeros((nv,nz))],np.c_[-np.eye(nv),np.zeros((nv,nz))],np.c_[np.zeros((nz,nv)),-np.eye(nz)]))
        rhs=np.r_[limits-vm,limits+vm,upper,-lower,np.zeros(nz)];A.append(linear);b.append(rhs);cones.append(clarabel.NonnegativeConeT(len(rhs)))
        settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=180
        settings.tol_gap_abs=settings.tol_gap_rel=settings.tol_feas=1e-10
        fit=clarabel.DefaultSolver(sparse.diags(np.r_[np.full(nv,2e-7),np.full(nz,2.)],format='csc'),np.zeros(nv+nz),
            sparse.csc_matrix(np.vstack(A)),np.concatenate(b),cones,settings).solve()
        status=str(fit.status);row=dict(iteration=it,status=status,merit=merit,maximum_position_excess_mm=maxpos,physical_scaled_excess=maxphys,radius=radius);history.append(row)
        if status not in ('Solved','AlmostSolved','InsufficientProgress'):break
        delta=np.zeros(n*6);delta[active]=np.asarray(fit.x[:nv]);delta=delta.reshape(n,6);best=None
        for fraction in (1.,.5,.25,.125,.0625,.03125,.015625,.0078125):
            trial=q.copy();trial[:,ids]+=fraction*delta;score=evaluate(trial)[3]
            if score<merit-1e-11 or score<=1e-8:best=trial,score,fraction;break
        row['accepted']=best is not None;row['linearized_elastic_norm_mm']=float(np.linalg.norm(fit.x[nv:]));row['delta_inf_rad']=float(np.abs(delta).max())
        if it%25==0:print('COMMON_SUM_CONIC',row,flush=True)
        if best is None:
            radius*=.5
            if radius<1e-7:break
            continue
        q,score,fraction=best;ratio=(merit-score)/max(merit-row['linearized_elastic_norm_mm'],1e-20)
        if ratio<.25:radius=max(radius*.5,1e-7)
        elif ratio>.75 and np.max(np.abs(delta))>.8*radius:radius=min(2*radius,trust_radius)
    return q,dict(history=history,iterations=len(history),raw_targets_modified=False,acceptance_slack_introduced=False)
