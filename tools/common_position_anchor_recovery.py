"""One shared position-only recovery: continuous full-limit anchors + reseeding.

Raw targets and timestamps are immutable. Anchors are posture guides, never new
Cartesian supervision. Executable states retain all original temporal bounds.
"""
import time
import numpy as np
from scipy.optimize import least_squares
from tools.final_common_position_solver import CommonPositionSolver
from tools.common_framewise_reachability_oracle import FramewiseReachabilityOracle
from tools.common_g1_position_bounds import outer_enclosures,residual_lower_bounds


class AnchorRecoveryPositionSolver(CommonPositionSolver):
    def __init__(self,g1,collision_model,config,natural_q,recovery_config):
        super().__init__(g1,collision_model,config,natural_q)
        self.recovery=dict(recovery_config)
        self.oracle=FramewiseReachabilityOracle(g1,collision_model,{'position_tolerance_m':self.tolerance,'seed_count':128,'maximum_evaluations_per_seed':180},natural_q)
        g1.assign(natural_q);self.enclosures=outer_enclosures(g1)

    def optimize(self,target,hands,seed,lower,upper,previous,predicted,guide,pairs,max_nfev):
        cached=None;value=None
        def compute(q):
            nonlocal cached,value
            if cached is not None and np.array_equal(q,cached):return value
            pos,jac=self.pose_jacobian(q,hands)
            residual=[1000*(pos-target).reshape(-1)];matrices=[1000*jac]
            if pairs:
                ds,js=self.clearance_values(pairs);active=ds<self.clearance
                residual.append(2000*np.minimum(ds-self.clearance,0));matrices.append(2000*js*active[:,None])
            for weight,ref in ((self.recovery['executable_guide_weight'],guide),(.03,previous),(.02,predicted),(.002,self.natural)):
                residual.append(weight*(q-ref));matrices.append(weight*np.eye(14))
            cached=q.copy();value=(np.concatenate(residual),np.vstack(matrices));return value
        result=least_squares(lambda q:compute(q)[0],np.clip(seed,lower+1e-10,upper-1e-10),jac=lambda q:compute(q)[1],
                            bounds=(lower,upper),max_nfev=max_nfev,ftol=1e-8,xtol=1e-8,gtol=1e-8)
        self.calls+=result.nfev
        return result.x,{'nfev':int(result.nfev),'optimizer_status':int(result.status)}

    def anchor(self,target,hands,previous,predicted):
        seeds=[previous,predicted,self.natural,*self.oracle.seeds[3:self.recovery['anchor_seed_budget']]]
        certificate=bool(residual_lower_bounds(target,self.enclosures).max()>self.tolerance+1e-8)
        candidates=[];best=None;bestkey=None
        for seed_id,seed in enumerate(seeds):
            if seed_id>=3 and (certificate or (bestkey is not None and not bestkey[0] and not bestkey[1])):break
            q,info=self.oracle.fit(target,hands,seed,reference=previous,posture_weight=self.recovery['anchor_posture_weight'])
            error=info['maximum_residual_m'];step=float(np.linalg.norm(q-previous))
            optimistic=(False,error>self.tolerance,step if error<=self.tolerance else error,step)
            if bestkey is not None and optimistic>=bestkey:
                candidates.append({'seed_id':seed_id,'q':q.tolist(),'contacts':None,
                                   'geometry_evaluated':False,'skip_reason':'PROVABLY_DOMINATED_EVEN_IF_COLLISION_FREE',
                                   'step_from_previous_rad':step,**info})
                continue
            contacts=self.contacts(q,hands)
            key=(bool(contacts),error>self.tolerance,step if error<=self.tolerance else error,step)
            candidates.append({'seed_id':seed_id,'q':q.tolist(),'contacts':contacts,'step_from_previous_rad':step,**info})
            if bestkey is None or key<bestkey:bestkey=key;best=q.copy()
        return best,{'candidates':candidates,'certified_outside_outer_workspace':certificate,'selected_key':bestkey}

    def solve(self,targets,hand_states,timestamps,initial_q):
        targets=np.asarray(targets,dtype=float).copy();hands=np.asarray(hand_states,dtype=float).copy()
        n=len(targets);dt=float(np.median(np.diff(timestamps)));started=time.monotonic()
        guide=np.empty((n,14));anchor_reports=[];previous=np.asarray(initial_q).copy();predicted=previous.copy()
        for frame in range(n):
            guide[frame],record=self.anchor(targets[frame],hands[frame],previous,predicted)
            predicted=2*guide[frame]-previous;previous=guide[frame]
            anchor_reports.append({'frame':frame,**record})
            if frame%150==0:print('anchor guide',frame,'/',n,flush=True)
        q=np.empty((n,14));q[0]=initial_q;reports=[];recoveries=[]
        for frame in range(1,n):
            value,report=self.realize_frame(targets[frame],hands[frame],q[frame-1],q[max(0,frame-2)],guide[frame],dt,True)
            error=np.linalg.norm(self.pose_jacobian(value,hands[frame])[0]-targets[frame],axis=1).max()
            if error>self.tolerance and residual_lower_bounds(targets[frame],self.enclosures).max()<=self.tolerance+1e-8:
                fresh,anchor_record=self.anchor(targets[frame],hands[frame],q[frame-1],2*q[frame-1]-q[max(0,frame-2)])
                retry,retry_report=self.realize_frame(targets[frame],hands[frame],q[frame-1],q[max(0,frame-2)],fresh,dt,True)
                retry_error=np.linalg.norm(self.pose_jacobian(retry,hands[frame])[0]-targets[frame],axis=1).max()
                old_bad=bool(self.contacts(value,hands[frame]));new_bad=bool(self.contacts(retry,hands[frame]))
                if (new_bad,retry_error)<(old_bad,error):value=retry;report=retry_report
                recoveries.append({'frame':frame,'initial_residual_m':float(error),'retry_residual_m':float(retry_error),'anchor':anchor_record})
            q[frame]=value;reports.append({'frame':frame,**report})
            if frame%150==0:print('recovered forward',frame,'/',n,flush=True)
        return q,{'runtime_s':time.monotonic()-started,'function_evaluations':self.calls,'frame_reports':reports,
                  'anchor_reports':anchor_reports,'framewise_recovery_attempts':recoveries,'posture_guide':guide}
