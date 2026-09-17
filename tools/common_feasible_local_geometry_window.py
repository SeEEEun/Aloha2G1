"""Keep physical and Cartesian constraints feasible during collision search."""
import numpy as np
from scipy.optimize import minimize,LinearConstraint,NonlinearConstraint,Bounds,BFGS
from scipy.sparse import csr_matrix

def solve(solver,targets,hands,initial,allowances,dt,arm,pairs,max_iterations=500):
    n=len(initial);ids=np.arange(arm*7,arm*7+6);base=initial[:,ids].copy();size=6*n;cfg=solver.config
    active=np.ones((n,6),bool);active[:2]=False;active[-2:]=False;active=active.ravel();fixed=base.ravel()
    d1=np.kron(np.diff(np.eye(n),axis=0),np.eye(6));d2=np.kron(np.diff(np.eye(n),n=2,axis=0),np.eye(6))
    step=min(cfg['maximum_joint_step_rad'],cfg['maximum_velocity_rad_s']*dt);acc=cfg['maximum_acceleration_rad_s2']*dt**2
    matrix=np.vstack((d1,d2));offset=matrix[:,~active]@fixed[~active];limit=np.r_[np.full(len(d1),step),np.full(len(d2),acc)]
    linear=LinearConstraint(csr_matrix(matrix[:,active]),-limit-offset,limit-offset,keep_feasible=True)
    def expand(x):v=fixed.copy();v[active]=x;return v
    cached=None;result=None;history=[]
    def evaluate(x):
        nonlocal cached,result
        if cached is not None and np.array_equal(cached,x):return result
        allq=expand(x);q=initial.copy();q[:,ids]=allq.reshape(n,6);pv=[];pj=[];cost=0.;grad=np.zeros(size)
        for f in range(n):
            pos,jac=solver.pose_jacobian(q[f],hands[f]);e=pos[arm]-targets[f,arm]
            pv.append(10000*(allowances[f,arm]**2-e@e));row=np.zeros(size);row[f*6:f*6+6]=-20000*e@jac[arm*3:arm*3+3,ids];pj.append(row)
            local_pairs=pairs.get(f,[]) if isinstance(pairs,dict) else pairs
            ds,js=solver.clearance_values(local_pairs);bad=np.minimum(ds-solver.clearance,0.)
            cost+=1e4*bad@bad;grad[f*6:f*6+6]+=2e4*bad@js[:,ids]
        delta=allq-fixed;vel=d1@allq;cost+=1e-7*(delta@delta)+1e-6*(vel@vel);grad+=2e-7*delta+2e-6*d1.T@vel
        result=(cost,grad[active],np.array(pv),csr_matrix(np.array(pj)[:,active]));cached=x.copy();return result
    def callback(x,state):
        if state.nit%25==0:
            history.append(dict(iteration=int(state.nit),objective=float(state.fun),constraint_violation=float(state.constr_violation)))
            print('KEEP_FEASIBLE_PROGRESS',history[-1],flush=True)
        return False
    assert evaluate(fixed[active])[2].min()>=0,'Feasible-only search requires valid initial raw/closest-feasible positions'
    fit=minimize(lambda x:evaluate(x)[0],fixed[active],jac=lambda x:evaluate(x)[1],hess=BFGS(),method='trust-constr',
        bounds=Bounds(np.tile(solver.lower[ids],n)[active],np.tile(solver.upper[ids],n)[active],keep_feasible=True),
        constraints=[linear,NonlinearConstraint(lambda x:evaluate(x)[2],0,np.inf,jac=lambda x:evaluate(x)[3],hess=BFGS(),keep_feasible=True)],
        callback=callback,options=dict(maxiter=max_iterations,gtol=1e-8,xtol=1e-10,barrier_tol=1e-10,sparse_jacobian=True,verbose=0))
    q=initial.copy();q[:,ids]=expand(fit.x).reshape(n,6)
    return q,dict(success=bool(fit.success),message=str(fit.message),iterations=int(fit.nit),cost=float(fit.fun),minimum_position_constraint=float(evaluate(fit.x)[2].min()),history=history)

