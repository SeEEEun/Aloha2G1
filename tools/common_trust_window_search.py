"""Independent trust-region hard-constraint retry for necessary arm windows."""
import numpy as np
from scipy.optimize import minimize,LinearConstraint,NonlinearConstraint,Bounds,BFGS
from scipy.sparse import csr_matrix


def solve_window(solver,targets,hands,initial,allowances,dt,arm,max_iterations=350):
    n=len(initial);indices=np.arange(arm*7,arm*7+6);reference=initial[:,indices].copy();cfg=solver.config
    lower=solver.lower[indices];upper=solver.upper[indices]
    d1=np.kron(np.diff(np.eye(n),axis=0),np.eye(6));d2=np.kron(np.diff(np.eye(n),n=2,axis=0),np.eye(6))
    step=min(cfg['maximum_joint_step_rad'],cfg['maximum_velocity_rad_s']*dt)-1e-6
    acc=cfg['maximum_acceleration_rad_s2']*dt*dt-1e-6
    linear=LinearConstraint(csr_matrix(np.vstack((d1,d2))),np.r_[np.full(len(d1),-step),np.full(len(d2),-acc)],np.r_[np.full(len(d1),step),np.full(len(d2),acc)])
    cached=None;values=None
    def nonlinear(x):
        nonlocal cached,values
        if cached is not None and np.array_equal(cached,x):return values
        q=initial.copy();q[:,indices]=x.reshape(n,6);errors=[];jac=[]
        for f in range(n):
            p,j=solver.pose_jacobian(q[f],hands[f]);e=p[arm]-targets[f,arm]
            errors.append(10000*((allowances[f,arm]-1e-6)**2-e@e))
            row=np.zeros(n*6);row[f*6:(f+1)*6]=-20000*e@j[arm*3:arm*3+3,indices];jac.append(row)
        d=(d1@x).reshape(n-1,6)
        for f in range(n-1):
            errors.append((cfg['maximum_step_norm_rad']-1e-6)**2-d[f]@d[f])
            row=np.zeros(n*6);row[f*6:(f+1)*6]=2*d[f];row[(f+1)*6:(f+2)*6]=-2*d[f];jac.append(row)
        cached=x.copy();values=np.array(errors),csr_matrix(np.array(jac));return values
    matrix=.002*d1.T@d1+2e-7*np.eye(n*6)
    def objective(x):
        d=d1@x;delta=x-reference.ravel()
        return .001*d@d+1e-7*delta@delta,.002*d1.T@d+2e-7*delta
    fit=minimize(lambda x:objective(x)[0],reference.ravel(),jac=lambda x:objective(x)[1],hess=lambda x:csr_matrix(matrix),
        method='trust-constr',bounds=Bounds(np.tile(lower,n),np.tile(upper,n)),
        constraints=[linear,NonlinearConstraint(lambda x:nonlinear(x)[0],0,np.inf,jac=lambda x:nonlinear(x)[1],hess=BFGS())],
        options={'maxiter':max_iterations,'gtol':1e-9,'xtol':1e-10,'barrier_tol':1e-10,'sparse_jacobian':True,'verbose':0})
    q=initial.copy();q[:,indices]=fit.x.reshape(n,6)
    error=[]
    for f in range(n):error.append(np.linalg.norm(solver.pose_jacobian(q[f],hands[f])[0][arm]-targets[f,arm]))
    delta=np.diff(q[:,indices],axis=0);accel=np.diff(q[:,indices],n=2,axis=0)/dt**2
    actual_step=min(cfg['maximum_joint_step_rad'],cfg['maximum_velocity_rad_s']*dt)
    passed=bool(np.all(np.array(error)<=allowances[:,arm]) and np.max(np.abs(delta))<=actual_step+1e-8 and
        np.max(np.linalg.norm(delta,axis=1))<=cfg['maximum_step_norm_rad']+1e-8 and np.max(np.abs(accel))<=cfg['maximum_acceleration_rad_s2']+1e-6)
    return q,dict(optimizer_success=bool(fit.success),message=str(fit.message),iterations=int(fit.nit),
        maximum_allowance_excess_mm=float(np.max(np.array(error)-allowances[:,arm])*1000),
        maximum_step_norm_rad=float(np.max(np.linalg.norm(delta,axis=1))),maximum_scalar_step=float(np.max(np.abs(delta))),
        maximum_acceleration=float(np.max(np.abs(accel))),constraint_violation=float(fit.constr_violation),relaxed_necessary_constraints_pass=passed)
