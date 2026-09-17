"""Common numerical building blocks. No representation or episode inputs."""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix, vstack, eye, kron, diags
from tools.final_common_position_solver import CommonPositionSolver


def preparation_path(start, end, intervals):
    u=np.linspace(0.,1.,intervals+1)
    blend=10*u**3-15*u**4+6*u**5
    q=start[None,:]+blend[:,None]*(end-start)[None,:]
    q[0]=start; q[-1]=end
    return q


def preparation_duration(start,end,dt,config):
    """Minimum time for the declared rest-to-rest quintic straight joint path.

    Not a globally time-optimal motion-planning claim. Analytical derivative
    envelopes and exact sampled step bounds are both enforced, with no padding.
    """
    delta=np.abs(end-start)
    duration=max(1.875*delta.max()/config['maximum_velocity_rad_s'],
                 np.sqrt((10/np.sqrt(3))*delta.max()/config['maximum_acceleration_rad_s2']),
                 1.875*np.linalg.norm(delta)*dt/config['maximum_step_norm_rad'],
                 1.875*delta.max()*dt/config['maximum_joint_step_rad'])
    n=max(1,int(np.ceil(duration/dt)))
    return n,preparation_path(start,end,n),float(duration)


def temporal_metrics(q,dt,config):
    d=np.diff(q,axis=0); dd=np.diff(q,n=2,axis=0)
    velocity=float(np.max(np.abs(d))/dt) if len(d) else 0.
    acceleration=float(np.max(np.abs(dd))/dt**2) if len(dd) else 0.
    norm=float(np.max(np.linalg.norm(d,axis=1))) if len(d) else 0.
    scalar=float(np.max(np.abs(d))) if len(d) else 0.
    return dict(maximum_velocity_rad_s=velocity,maximum_acceleration_rad_s2=acceleration,
                maximum_step_norm_rad=norm,maximum_scalar_step_rad=scalar,
                pass_temporal=bool(velocity<=config['maximum_velocity_rad_s']+1e-7 and
                 acceleration<=config['maximum_acceleration_rad_s2']+1e-5 and
                 norm<=config['maximum_step_norm_rad']+1e-7 and scalar<=config['maximum_joint_step_rad']+1e-7))


def propagated_candidate(oracle,targets,hands,seed,reverse=False,weight=.001):
    q=np.empty((len(targets),14));previous=seed.copy()
    sequence=range(len(targets)-1,-1,-1) if reverse else range(len(targets))
    for frame in sequence:
        previous,_=oracle.fit(targets[frame],hands[frame],previous,reference=previous,
                             posture_weight=weight,max_evaluations=100)
        q[frame]=previous
    return q


def refine_trajectory(solver,targets,hands,initial,weight,max_evaluations=100):
    """Bounded sparse whole-trajectory kinematic continuation candidate.

    This generator does not declare collision or Cartesian acceptance. Every
    generated trajectory must independently pass the unchanged physical rules.
    Raw targets are never overwritten. Acceleration regularization is not a
    substitute for hard temporal verification.
    """
    n=len(initial); size=n*14
    d1=kron(diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n)),eye(14),format='csr')
    d2=kron(diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n)),eye(14),format='csr')
    rows=np.repeat(np.arange(n*6),14)
    cols=np.concatenate([np.tile(np.arange(f*14,(f+1)*14),6) for f in range(n)])
    cached=None;result=None
    def compute(x):
        nonlocal cached,result
        if cached is not None and np.array_equal(cached,x):return result
        q=x.reshape(n,14);pos=[];jac=[]
        for f in range(n):
            p,j=solver.pose_jacobian(q[f],hands[f]);pos.append(p);jac.append(j)
        residual=np.r_[(np.array(pos)-targets).ravel(),weight*(d1@x),3*weight*(d2@x),1e-6*(x-initial.ravel())]
        matrix=vstack([coo_matrix((np.array(jac).ravel(),(rows,cols)),shape=(n*6,size)),weight*d1,3*weight*d2,1e-6*eye(size)],format='csr')
        cached=x.copy();result=(residual,matrix);return result
    fit=least_squares(lambda x:compute(x)[0],initial.ravel(),jac=lambda x:compute(x)[1],
        bounds=(np.tile(solver.lower,n),np.tile(solver.upper,n)),max_nfev=max_evaluations,
        ftol=1e-8,xtol=1e-8,gtol=1e-8,tr_solver='lsmr',tr_options={'maxiter':120})
    return fit.x.reshape(n,14),dict(nfev=int(fit.nfev),status=int(fit.status),cost=float(fit.cost))
