"""Common bounded closest-feasible constraint realization, with raw targets fixed.

Elastic variables measure residual gate violations during search only. They
never change acceptance or raw targets. Hard joints / dynamics and geometry
search planes remain unrelaxed; complete actual geometry is checked afterward.
"""
import numpy as np
from scipy.optimize import linprog

def solve(solver,targets,hands,initial,allowances,dt,arm,goals,trust_radius=.0005,max_iterations=200):
    q=initial.copy();n=len(q);ids=np.arange(arm*7,arm*7+6)
    active=np.ones((n,6),bool);active[:2]=active[-2:]=False;active=active.ravel();nv=int(active.sum())
    d1=np.kron(np.diff(np.eye(n),axis=0),np.eye(6));d2=np.kron(np.diff(np.eye(n),n=2,axis=0),np.eye(6))
    cfg=solver.config;step=min(cfg['maximum_joint_step_rad'],cfg['maximum_velocity_rad_s']*dt)
    matrix=np.vstack((d1,d2));limits=np.r_[np.full(len(d1),step),np.full(len(d2),cfg['maximum_acceleration_rad_s2']*dt**2)]
    history=[];radius=trust_radius
    def evaluate(v):
        aa=[];bb=[];types=[]
        for f in range(2,n-2):
            pos,jac=solver.pose_jacobian(v[f],hands[f]);e=pos[arm]-targets[f,arm];length=np.linalg.norm(e)
            row=np.zeros(n*6);row[f*6:f*6+6]=1e3*(e/max(length,1e-30))@jac[arm*3:arm*3+3,ids]
            aa.append(row[active]);bb.append(1e3*(allowances[f,arm]-length-1e-10));types.append(True)
            if f in goals:
                ds,js=solver.clearance_values(goals[f])
                for dist,j in zip(ds,js):
                    row=np.zeros(n*6);row[f*6:f*6+6]=-1e3*j[ids]
                    aa.append(row[active]);bb.append(1e3*(dist-solver.clearance-1e-10));types.append(False)
        return np.array(aa),np.array(bb),np.array(types)
    for it in range(max_iterations):
        base=q[:,ids].ravel();aa,bb,elastic=evaluate(q);metric=float(max(0.,-bb.min()))
        vm=matrix@base
        A=np.vstack((np.c_[aa,-elastic.astype(float)],np.c_[matrix[:,active],np.zeros(len(matrix))],np.c_[-matrix[:,active],np.zeros(len(matrix))]))
        b=np.r_[bb,limits-vm,limits+vm]
        lower=np.maximum(np.tile(solver.lower[ids],n)[active]-base[active],-radius)
        upper=np.minimum(np.tile(solver.upper[ids],n)[active]-base[active],radius)
        opt={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9}
        fit=linprog(np.r_[np.zeros(nv),1.],A_ub=A,b_ub=b,bounds=[*zip(lower,upper),(0,None)],method='highs',options=opt)
        if not fit.success:
            history.append(dict(iteration=it,lp_success=False,message=str(fit.message)));break
        # Deterministic lexicographic minimum-change tie break.
        Az=np.c_[A,np.zeros(len(A))]
        Az=np.vstack((Az,np.r_[np.zeros(nv),1.,0.],np.c_[np.eye(nv),np.zeros(nv),-np.ones(nv)],np.c_[-np.eye(nv),np.zeros(nv),-np.ones(nv)]))
        bz=np.r_[b,fit.x[-1]+1e-10,np.zeros(2*nv)]
        tied=linprog(np.r_[np.zeros(nv+1),1.],A_ub=Az,b_ub=bz,bounds=[*zip(lower,upper),(0,None),(0,radius)],method='highs',options=opt)
        delta=np.zeros(n*6);delta[active]=(tied.x[:nv] if tied.success else fit.x[:nv]);delta=delta.reshape(n,6)
        best=None
        for fraction in (1.,.5,.25,.125,.0625,.03125,.015625,.0078125):
            trial=q.copy();trial[:,ids]+=fraction*delta
            _,values,_=evaluate(trial);score=float(max(0.,-values.min()))
            if score<metric-1e-12 or score<=1e-10:
                best=(trial,score,fraction);break
        row=dict(iteration=it,lp_success=True,old_maximum_violation_mm=metric,linearized_position_violation_mm=float(fit.x[-1]),
                 trust_radius_rad=radius,accepted=best is not None)
        history.append(row)
        if it%10==0:print('COMMON_ELASTIC_REALIZATION',row,flush=True)
        if best is None:
            radius*=.5
            if radius<1e-7:break
            continue
        q,score,fraction=best;row.update(new_maximum_violation_mm=score,line_fraction=fraction)
        if score<=1e-10:break
    return q,dict(iterations=len(history),history=history,raw_targets_modified=False,acceptance_requires_independent_requalification=True)
