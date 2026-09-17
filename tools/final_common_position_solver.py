"""Bounded method-blind sequential wrist-position solver using the actual G1.

No episode, representation, object, semantic phase or policy outcome is input.
Two propagated passes share limits, collision model and deterministic seeds.
"""
from __future__ import annotations

import time
from itertools import product
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.stats import qmc


class CommonPositionSolver:
    def __init__(self, g1, collision_model, config, natural_q):
        self.g1, self.collision = g1, collision_model
        self.config = dict(config)
        self.natural = np.asarray(natural_q,dtype=float)
        self.lower = g1.arm_limits[:,0]+1e-7
        self.upper = g1.arm_limits[:,1]-1e-7
        self.tolerance = float(config["position_tolerance_m"])
        self.clearance = float(config["collision_clearance_m"])
        self.tracked = set()
        self.calls = 0

    def pose_jacobian(self, q, hands):
        self.g1.assign(q,*hands)
        pos, jac = [], []
        for side in ("left","right"):
            body = self.g1.wrist_ids[side]
            pos.append(self.g1.data.xpos[body].copy())
            jp = np.zeros((3,self.g1.model.nv))
            mujoco.mj_jacBody(self.g1.model,self.g1.data,jp,None,body)
            jac.append(jp[:,self.g1.arm_dof_ids])
        return np.asarray(pos), np.vstack(jac)

    def contacts(self,q,hands):
        return self.collision._records(q,*hands)

    def clearance_values(self, pairs):
        ds, js = [], []
        model,data = self.g1.model,self.g1.data
        for pair in pairs:
            segment = np.zeros(6)
            d = float(mujoco.mj_geomDistance(model,data,*pair,0.02,segment))
            ds.append(d)
            derivative = np.zeros(14)
            if d < 0.02 and abs(d)>1e-10:
                normal = (segment[3:]-segment[:3])/d
                matrices=[]
                for geom,p in zip(pair,(segment[:3],segment[3:])):
                    jp=np.zeros((3,model.nv))
                    mujoco.mj_jac(model,data,jp,None,p,int(model.geom_bodyid[geom]))
                    matrices.append(jp[:,self.g1.arm_dof_ids])
                derivative = normal @ (matrices[1]-matrices[0])
            js.append(derivative)
        return np.asarray(ds),np.asarray(js).reshape(-1,14)

    def optimize(self,target,hands,seed,lower,upper,previous,predicted,guide,pairs,max_nfev):
        cached_q=None; cached=None
        # Cartesian objective dominates posture preferences. Collision penalty
        # is active at actual signed geometry distances and verified afterward.
        references=[(0.03,previous),(0.02,predicted),(0.01,guide),(0.002,self.natural)]
        def evaluate(q):
            nonlocal cached_q,cached
            if cached_q is not None and np.array_equal(q,cached_q): return cached
            pos,jac = self.pose_jacobian(q,hands)
            residual=[1000*(pos-target).reshape(-1)]
            matrices=[1000*jac]
            if pairs:
                distances,derivatives=self.clearance_values(pairs)
                active=distances<self.clearance
                residual.append(2000*np.minimum(distances-self.clearance,0))
                matrices.append(2000*derivatives*active[:,None])
            for weight,ref in references:
                residual.append(weight*(q-ref)); matrices.append(weight*np.eye(14))
            cached_q=q.copy(); cached=(np.concatenate(residual),np.vstack(matrices))
            return cached
        result=least_squares(lambda q:evaluate(q)[0],np.clip(seed,lower+1e-10,upper-1e-10),jac=lambda q:evaluate(q)[1],bounds=(lower,upper),max_nfev=max_nfev,ftol=1e-8,xtol=1e-8,gtol=1e-8)
        self.calls+=result.nfev
        return result.x, {"nfev":int(result.nfev),"optimizer_status":int(result.status)}

    def realize_frame(self,target,hands,previous,previous_previous,guide,dt,temporal):
        lower,upper=self.lower.copy(),self.upper.copy()
        if temporal:
            step=min(self.config["maximum_joint_step_rad"],self.config["maximum_velocity_rad_s"]*dt)-1e-5
            lower=np.maximum(lower,previous-step); upper=np.minimum(upper,previous+step)
            # Conservative acceleration-compatible stopping distance near hard
            # bounds prevents an empty next-frame intersection.
            predicted=2*previous-previous_previous
            astep=self.config["maximum_acceleration_rad_s2"]*dt*dt-1e-5
            lower=np.maximum(lower,predicted-astep); upper=np.minimum(upper,predicted+astep)
        else:
            predicted=previous
        if np.any(lower>=upper):
            return previous.copy(), {"bounds_infeasible":True,"candidates":[]}
        offset=np.zeros(14); offset[[1,2,3,8,9,10]]=[0.2,-0.3,0.25,-0.2,0.3,0.25]
        seeds=[previous,predicted,guide,self.natural+offset]
        candidates=[]; best=None; bestkey=None
        for seed_index,seed in enumerate(seeds[:self.config["candidate_count_per_frame"]]):
            value=np.clip(seed,lower+1e-10,upper-1e-10)
            local_pairs=set(self.tracked)
            # Limit tracked set to near pairs at this seed, so departed
            # contacts do not burden every later frame.
            self.g1.assign(value,*hands)
            local_pairs={p for p in local_pairs if mujoco.mj_geomDistance(self.g1.model,self.g1.data,*p,0.02,None)<0.02}
            for discovery in range(self.config["collision_discovery_passes_per_candidate"]):
                initial_contacts=self.contacts(value,hands)
                local_pairs.update(tuple(r["geom_pair"]) for r in initial_contacts)
                value,info=self.optimize(target,hands,value,lower,upper,previous,predicted,guide,sorted(local_pairs),self.config["maximum_function_evaluations_per_candidate"])
                if temporal:
                    delta=value-previous
                    norm=np.linalg.norm(delta)
                    if norm>self.config["maximum_step_norm_rad"]:
                        value=previous+delta*(self.config["maximum_step_norm_rad"]/norm)
                contacts=self.contacts(value,hands)
                self.tracked.update(tuple(r["geom_pair"]) for r in contacts)
                residual=np.linalg.norm(self.pose_jacobian(value,hands)[0]-target,axis=1)
                acceleration_ok=not temporal or np.max(np.abs(value-2*previous+previous_previous))<=self.config["maximum_acceleration_rad_s2"]*dt*dt+1e-8
                key=(float(np.max(residual))>self.tolerance, len(contacts)>0, not acceleration_ok, float(np.max(residual)) if np.max(residual)>self.tolerance else 0, len(contacts), float(np.linalg.norm(value-previous)),float(np.linalg.norm(value-self.natural)))
                record={"seed_index":seed_index,"discovery_pass":discovery,"q":value.tolist(),"position_residual_m":residual.tolist(),"contacts":contacts,"acceleration_valid":acceleration_ok,**info}
                candidates.append(record)
                if bestkey is None or key<bestkey: bestkey=key; best=value.copy()
                if not contacts: break
            if not bestkey[0] and not bestkey[1] and not bestkey[2]: break
        return best,{"bounds_infeasible":False,"candidates":candidates,"selected_key":bestkey}

    def solve(self,targets,hand_states,timestamps,initial_q):
        targets=np.asarray(targets,dtype=float).copy()
        hand_states=np.asarray(hand_states,dtype=float)
        initial_q=np.asarray(initial_q,dtype=float)
        n=len(targets); dt=float(np.median(np.diff(timestamps)))
        assert targets.shape==(n,2,3) and hand_states.shape==(n,2,7)
        assert np.isfinite(targets).all() and np.isfinite(hand_states).all()
        started=time.monotonic(); guide=np.repeat(initial_q[None,:],n,axis=0)
        reports=[]
        # Reverse guide is not an executable trajectory. It provides future
        # collision-aware posture seeds to the subsequent bounded forward pass.
        previous=initial_q.copy()
        for frame in range(n-1,-1,-1):
            value,report=self.realize_frame(targets[frame],hand_states[frame],previous,previous,previous,dt,False)
            guide[frame]=value; previous=value
            if frame%150==0: print(f"reverse guide frame {frame}/{n}",flush=True)
        q=np.empty((n,14)); q[0]=initial_q
        for frame in range(1,n):
            value,report=self.realize_frame(targets[frame],hand_states[frame],q[frame-1],q[max(0,frame-2)],guide[frame],dt,True)
            q[frame]=value
            report["frame"]=frame
            reports.append(report)
            if frame%150==0: print(f"forward trajectory frame {frame}/{n}",flush=True)
        return q,{"runtime_s":time.monotonic()-started,"function_evaluations":self.calls,"frame_reports":reports,"reverse_guide":guide}

    def witness(self,target,hands):
        """Bounded full-limit multi-seed residual audit, not a global proof."""
        samples=qmc.Halton(14,scramble=False).random(self.config["global_witness_seed_count"]-2)
        seeds=[self.natural,self.g1.stand_qpos[self.g1.arm_qpos_ids],*(self.lower+(self.upper-self.lower)*samples)]
        rows=[]
        for seed in seeds:
            value,info=self.optimize(target,hands,seed,self.lower,self.upper,self.natural,self.natural,self.natural,[],self.config["global_witness_evaluations_per_seed"])
            residual=np.linalg.norm(self.pose_jacobian(value,hands)[0]-target,axis=1)
            rows.append({"q":value.tolist(),"position_residual_m":residual.tolist(),"contacts":self.contacts(value,hands),**info})
        return {"seeds":rows,"best_maximum_wrist_residual_m":min(max(r["position_residual_m"]) for r in rows),"global_infeasibility_proven":False}
