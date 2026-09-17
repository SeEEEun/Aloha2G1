"""Bounded hard-constrained arm-window position search, method blind.

Arm-only step norms relax the bilateral constraint and collisions are omitted
for this necessary-condition search. A pass is therefore only a kinematic
witness; a finite failure is not a global impossibility certificate.
"""
import numpy as np
from scipy.optimize import minimize,LinearConstraint


def solve_window(solver,targets,hands,initial,allowances,dt,arm,max_iterations=400):
    n=len(initial);indices=np.arange(arm*7,arm*7+6);reference=initial[:,indices].copy()
    lower=solver.lower[indices];upper=solver.upper[indices];cfg=solver.config
    d1=np.kron(np.diff(np.eye(n),axis=0),np.eye(6))
    d2=np.kron(np.diff(np.eye(n),n=2,axis=0),np.eye(6))
    step=min(cfg['maximum_joint_step_rad'],cfg['maximum_velocity_rad_s']*dt)
    acc=cfg['maximum_acceleration_rad_s2']*dt*dt
    linear=LinearConstraint(np.vstack((d1,d2)),np.r_[np.full(len(d1),-step),np.full(len(d2),-acc)],np.r_[np.full(len(d1),step),np.full(len(d2),acc)])
    cached=None;values=None
    def position(x):
        nonlocal cached,values
        if cached is not None and np.array_equal(cached,x):return values
        q=initial.copy();q[:,indices]=x.reshape(n,6);errors=[];jac=[]
        for f in range(n):
            p,j=solver.pose_jacobian(q[f],hands[f]);e=p[arm]-targets[f,arm]
            errors.append(10000*(allowances[f,arm]**2-e@e))
            row=np.zeros(n*6);row[f*6:(f+1)*6]=-20000*e@j[arm*3:arm*3+3,indices];jac.append(row)
        cached=x.copy();values=np.array(errors),np.array(jac);return values
    def norm_constraint(x):
        d=(d1@x).reshape(n-1,6)
        val=cfg['maximum_step_norm_rad']**2-np.sum(d*d,axis=1)
        jac=np.zeros((n-1,n*6))
        for f in range(n-1):jac[f,f*6:(f+1)*6]=2*d[f];jac[f,(f+1)*6:(f+2)*6]=-2*d[f]
        return val,jac
    def objective(x):
        d=d1@x;delta=x-reference.ravel()
        return .001*(d@d)+1e-7*(delta@delta),.002*d1.T@d+2e-7*delta
    fit=minimize(lambda x:objective(x)[0],reference.ravel(),jac=lambda x:objective(x)[1],
        method='SLSQP',bounds=list(zip(np.tile(lower,n),np.tile(upper,n))),
        constraints=[linear,{'type':'ineq','fun':lambda x:position(x)[0],'jac':lambda x:position(x)[1]},
                     {'type':'ineq','fun':lambda x:norm_constraint(x)[0],'jac':lambda x:norm_constraint(x)[1]}],
        options={'maxiter':max_iterations,'ftol':1e-12,'disp':False})
    q=initial.copy();q[:,indices]=fit.x.reshape(n,6)
    pv=position(fit.x)[0];nv=norm_constraint(fit.x)[0];d=d1@fit.x;a=d2@fit.x
    return q,dict(optimizer_success=bool(fit.success),message=str(fit.message),iterations=int(fit.nit),
        minimum_position_constraint=float(pv.min()),minimum_step_norm_constraint=float(nv.min()),
        maximum_scalar_step=float(np.abs(d).max()),maximum_acceleration=float(np.abs(a).max()/dt**2),
        relaxed_necessary_constraints_pass=bool(pv.min()>=-1e-9 and nv.min()>=-1e-9 and np.abs(d).max()<=step+1e-8 and np.abs(a).max()<=acc+1e-8))
