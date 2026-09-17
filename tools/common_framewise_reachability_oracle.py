"""Bounded, deterministic, frame-independent G1 position feasibility search.

No representation, episode, object, previous trajectory, timing, or outcome input.
FRAME_UNREACHABLE means no witness in the declared finite search, not a theorem.
"""
from itertools import product
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc


class FramewiseReachabilityOracle:
    def __init__(self,kinematics,collision,config,natural_q):
        self.g1=kinematics;self.collision=collision;self.config=dict(config)
        self.natural=np.asarray(natural_q,dtype=float)
        self.lower=self.g1.arm_limits[:,0]+1e-7
        self.upper=self.g1.arm_limits[:,1]-1e-7
        self.tolerance=float(config['position_tolerance_m'])
        seeds=[self.natural,self.g1.stand_qpos[self.g1.arm_qpos_ids],(self.lower+self.upper)/2]
        for a,b,c,d in product((-1.,1.),repeat=4):
            offset=np.zeros(14)
            offset[[0,1,2,3,7,8,9,10]]=[.5*a,.35*b,.65*c,.45*d,.5*a,-.35*b,-.65*c,.45*d]
            seeds.append(self.natural+offset)
        samples=qmc.Halton(14,scramble=False).random(config['seed_count']-len(seeds))
        seeds.extend(self.lower+(self.upper-self.lower)*samples)
        self.seeds=np.clip(np.asarray(seeds),self.lower+1e-10,self.upper-1e-10)

    def fit(self,target,hands,seed,max_evaluations=None,reference=None,posture_weight=0.):
        """Exact incoming position objective; optional common posture preference."""
        import mujoco
        cached=None;value=None
        def compute(q):
            nonlocal cached,value
            if cached is not None and np.array_equal(q,cached):return value
            self.g1.assign(q,*hands)
            positions=[];jac=[]
            for side in ('left','right'):
                body=self.g1.wrist_ids[side];positions.append(self.g1.data.xpos[body].copy())
                jp=np.zeros((3,self.g1.model.nv));mujoco.mj_jacBody(self.g1.model,self.g1.data,jp,None,body)
                jac.append(jp[:,self.g1.arm_dof_ids])
            residual=(np.asarray(positions)-target).reshape(-1)
            matrix=np.vstack(jac)
            if reference is not None and posture_weight:
                residual=np.r_[residual,posture_weight*(q-reference)]
                matrix=np.vstack((matrix,posture_weight*np.eye(14)))
            cached=q.copy();value=(residual,matrix)
            return value
        fit=least_squares(lambda q:compute(q)[0],np.clip(seed,self.lower+1e-10,self.upper-1e-10),jac=lambda q:compute(q)[1],
                          bounds=(self.lower,self.upper),max_nfev=max_evaluations or self.config['maximum_evaluations_per_seed'],
                          ftol=1e-11,xtol=1e-11,gtol=1e-11)
        error=np.linalg.norm(compute(fit.x)[0][:6].reshape(2,3),axis=1)
        return fit.x,{'residual_m':error.tolist(),'maximum_residual_m':float(error.max()),'nfev':int(fit.nfev),'optimizer_status':int(fit.status)}

    def solve(self,target,hands):
        target=np.asarray(target,dtype=float).copy();hands=np.asarray(hands,dtype=float).copy()
        if target.shape!=(2,3) or hands.shape!=(2,7) or not np.isfinite(target).all() or not np.isfinite(hands).all():
            raise ValueError('finite bilateral target [2,3] and hand state [2,7] required')
        started=time.monotonic();candidates=[];best=None;best_valid=None
        for seed_id,seed in enumerate(self.seeds):
            q,info=self.fit(target,hands,seed)
            margin=float(np.min(np.minimum(q-self.g1.arm_limits[:,0],self.g1.arm_limits[:,1]-q)))
            # Confirmation is deferred until a seed gives Cartesian acceptance,
            # or improves the correction upper bound. No proxy waiver is used.
            check=info['maximum_residual_m']<=self.tolerance or best is None or info['maximum_residual_m']<best['maximum_residual_m']
            records=self.collision.inspect(q,*hands) if check else None
            blocked=None if records is None else any(r['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for r in records)
            row={'seed_id':seed_id,'q':q.tolist(),'joint_limit_margin_rad':margin,'detailed_collision_records':records,
                 'geometry_checked':records is not None,'geometry_valid':None if blocked is None else not blocked,
                 'finite':bool(np.isfinite(q).all()),**info}
            candidates.append(row)
            if best is None or info['maximum_residual_m']<best['maximum_residual_m']:best=row
            if blocked is False and margin>=0 and row['finite']:
                if best_valid is None or info['maximum_residual_m']<best_valid['maximum_residual_m']:best_valid=row
                if info['maximum_residual_m']<=self.tolerance:
                    return {'classification':'FRAME_REACHABLE','best':best_valid,'best_kinematic':best,
                            'candidates':candidates,'seeds_tested':len(candidates),'runtime_s':time.monotonic()-started,
                            'global_unreachability_proven':False,'correction_to_satisfy_gate_upper_bound_m':0.}
        return {'classification':'FRAME_UNREACHABLE','interpretation':'NO_VALID_WITNESS_IN_BOUNDED_SEARCH; not globally exhaustive or a physical impossibility proof',
                'best':best_valid or best,'best_kinematic':best,'candidates':candidates,'seeds_tested':len(candidates),
                'runtime_s':time.monotonic()-started,'global_unreachability_proven':False,
                'correction_to_satisfy_gate_upper_bound_m':max(best_valid['maximum_residual_m']-self.tolerance,0.) if best_valid else None,
                'minimum_correction_certified_lower_bound_m':0.}
